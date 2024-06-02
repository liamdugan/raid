import itertools
import os
from copy import copy
from pathlib import Path

import fasttext.util
import nltk
import torch
import transformers
from sklearn.metrics.pairwise import cosine_similarity


class SynonymAttack:
    def __init__(self, N=0.5):
        # The percentage of words in the text to be replaced
        self.N = N
        # The probability threshold under which to ignore a BERT synonym
        self.bert_threshold = 0.0025
        # The total number of top k BERT candidates to consider
        self.num_candidates = 20
        # If a synonym's word embedding and the original word have
        # cosine similarity less than this threshold, the synonym is discarded
        self.cs_thresh = 0.5
        # Number of words to include on either side of the target when computing BERT synonyms
        self.bert_window_size = 20
        # Number of words to include on either side of the target when calculating POS tag
        self.pos_window_size = 4

        # Load BERT
        device = 0 if torch.cuda.is_available() else -1
        tok_kwargs = {"truncation": True}
        self.bert = transformers.pipeline(
            "fill-mask", model="bert-base-cased", device=device, tokenizer_kwargs=tok_kwargs
        )

        # Load FastText
        running_path = os.getcwd()
        resources_path = Path(os.path.dirname(os.path.abspath(__file__))) / "resources"
        os.chdir(resources_path)
        fasttext.util.download_model("en", if_exists="ignore")
        self.fasttext = fasttext.load_model("cc.en.300.bin")
        os.chdir(running_path)

        # Load NLTK Sentence tokenizer and word tokenizer
        self.sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sspan = lambda x: self.sent_tokenizer.span_tokenize(x)
        self.word_tokenizer = nltk.tokenize.NLTKWordTokenizer()
        self.wspan = lambda x: self.word_tokenizer.span_tokenize(x)

        # Load NLTK stopwords
        self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def embed(self, word):
        return self.fasttext[word].reshape(1, -1)

    def get_window(self, spans, i, width):
        # Return a span of tokens from i-w to i+w
        left = max(0, i - width)
        right = min(i + width, len(spans))
        return spans[left:right], i - left

    def get_synonym(self, i, toks, tag):
        # Our word of interest is at the ith position in the tokens
        word = toks[i]

        # If the word is non alphanumeric or is in the set of stopwords, return no synonyms
        if word.lower() in self.stopwords or not word.isalpha():
            return []

        # Get a window of tokens around the word where window[j] = toks[i] for BERT
        window, j = self.get_window(toks, i, self.bert_window_size * 2)
        window[j] = "[MASK]"

        # Get the top k BERT candidates for the word at that position:
        try:
            candidates = self.bert("".join(window), top_k=self.num_candidates)
        except transformers.pipelines.base.PipelineException:
            return []  # when window > 512 bert tokens, return no candidates

        # Retain candidates that are not the original word and are not sub-word tokens
        candidates = [c for c in candidates if c["token_str"] != word and c["token_str"].isalpha()]

        # Get a window of tokens around the word where window[j] = toks[i] for POS tagging
        window, j = self.get_window(toks, i, self.pos_window_size * 2)

        # Get POS tags for each of the candidate synonyms
        candidate_tags = []
        for c in candidates:
            window[j] = c["token_str"]
            candidate_tags.append(nltk.pos_tag(window)[j][1])

        # Retain only candidates that match the POS tag of the original token (and is not the original word)
        candidates = [c for c, t in zip(candidates, candidate_tags) if t == tag]

        # Get the FastText embedding for our original word
        e = self.embed(word)

        # Retain only candidates that have high cosine similarity to the original word
        candidates = [c for c in candidates if cosine_similarity(e, self.embed(c["token_str"])) >= self.cs_thresh]

        # Retain only candidates that are above a certain BERT probability
        candidates = [c for c in candidates if c["score"] >= self.bert_threshold]

        # From the remaining candidates, return the one with the maximum likelihood according to BERT
        return max(candidates, key=lambda x: x["score"], default=None)

    def attack(self, text):
        # Get spans for the words in the text from NLTK
        # e.g. "Hi my name is" -> [(0, 2), (3, 5), (6, 10), (11, 13)]
        word_spans = [(ws + s, we + s) for s, e in self.sspan(text) for ws, we in self.wspan(text[s:e])]

        # Expand the list of span tokens to include spans inbetween tokens
        # e.g. "Hi my name is" -> [(0, 2), (2, 3), (3, 5), (5, 6), (6, 10), (10, 11), (11, 13)]
        all_spans = list(itertools.pairwise(itertools.chain.from_iterable(word_spans)))
        toks = [text[s:e] for s, e in all_spans]

        # Get the POS tags from nltk for the sequence of word tokens
        tags = [tag for _, tag in nltk.pos_tag([text[s:e] for s, e in word_spans])]

        # Get candidate synonyms for every word in toks
        # (use verbosity of "error" to avoid annoying huggingface parallelization warnings)
        transformers.logging.set_verbosity_error()
        transformers.logging.captureWarnings(True)
        candidates = [
            (w, i, self.get_synonym(i, copy(toks), tag)) for (i, w), tag in zip(list(enumerate(toks))[::2], tags)
        ]
        transformers.logging.captureWarnings(False)
        transformers.logging.set_verbosity_warning()

        # Select the total number of words to alter as some percentage N of total word tokens
        # (with the maximum being the total number of words with valid synonyms)
        candidates = [x for x in candidates if x[2]]
        words_to_alter = min(int(len(word_spans) * self.N), len(candidates))

        # Sort candidates by their BERT score
        candidates = sorted(candidates, key=lambda x: x[2]["score"], reverse=True)

        # Select the words_to_alter best swaps according to BERT probability
        # (replace by order of appearance in the string)
        edits = []
        for w, i, synonym in sorted(candidates[:words_to_alter], key=lambda x: x[1]):
            toks[i] = synonym["token_str"]
            edits.append((len("".join(toks[:i])), len("".join(toks[: i + 1]))))

        return {"generation": "".join(toks), "num_edits": len(edits), "edits": edits}
