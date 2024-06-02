import itertools
import json
import os
import random
from pathlib import Path

import nltk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PerplexityMisspellingAttack:
    def __init__(self, N=0.5, model="gpt2"):
        self.N = N

        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Load NLTK
        self.sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sspan = lambda x: self.sent_tokenizer.span_tokenize(x)
        self.word_tokenizer = nltk.tokenize.NLTKWordTokenizer()
        self.wspan = lambda x: self.word_tokenizer.span_tokenize(x)

        # Get the directory of the currently executing script
        current_directory = Path(os.path.dirname(os.path.abspath(__file__)))

        # Load misspellings json file
        json_path = current_directory / "resources" / "misspelling3.json"
        with open(json_path, "r") as f:
            self.corrections = json.load(f)

    def get_nltk_span_tokens(self, text):
        return [(ws + s, we + s) for s, e in self.sspan(text) for ws, we in self.wspan(text[s:e])]

    def attack(self, text):
        # Get spans for the words in the text from NLTK
        # e.g. "Hi my name is" -> [(0, 2), (3, 5), (6, 10), (11, 13)]
        word_spans = self.get_nltk_span_tokens(text)

        # Expand the list of span tokens to include spans inbetween tokens
        # e.g. "Hi my name is" -> [(0, 2), (2, 3), (3, 5), (5, 6), (6, 10), (10, 11), (11, 13)]
        all_spans = list(itertools.pairwise(itertools.chain.from_iterable(word_spans)))
        toks = [text[s:e] for s, e in all_spans]

        # Get a list of all spans in the passage that have misspellings available
        candidate_spans = [(i, s, e) for i, (s, e) in enumerate(all_spans) if self.can_misspell(text[s:e])]

        # The total number of misspellings should be some percentage N of the total word spans
        # in the passage (with the maximum being the total number of candidates)
        words_to_alter = min(int(len(word_spans) * self.N), len(candidate_spans))

        # If no words to alter we return the text
        if words_to_alter == 0:
            return {"generation": text, "num_edits": 0, "edits": []}

        # If we're altering all candidates, no need to calculate perplexity, misspell all candidates
        if words_to_alter == len(candidate_spans):
            edits = []
            for i, s, e in candidate_spans:
                toks[i] = self.misspell(text[s:e])
                edits.append((len("".join(toks[:i])), len("".join(toks[: i + 1]))))

            return {"generation": "".join(toks), "num_edits": len(edits), "edits": edits}

        # Else, get the negative log likelihoods of the sequence
        nlls = self.calculate_perplexity(text)

        # Compute the start and end indices in the original text for each model token
        token_indices = list(itertools.accumulate([len(w) for w, _ in nlls]))
        token_spans = list(zip([0] + token_indices[:-1], token_indices))

        # For each candidate word, sum the negative log-likelihoods of all overlapping model tokens
        nll_tokens = []
        for _, start, end in candidate_spans:
            nll_tokens.append(sum([nlls[i][1] for i, (s, e) in enumerate(token_spans) if (start < e) and (end > s)]))

        # Sort all candidate words from most to least likely
        sorted_nlls = sorted(list(zip(candidate_spans, nll_tokens)), key=lambda x: x[1], reverse=True)

        # Remove all tokens with a negative log-likelihood of 0
        # NOTE: This happens when we have inputs longer than 1024 tokens. All tokens after max len get a nll of 0
        #       This is not ideal. We would like to use a sliding window to calculate nlls for all tokens
        #       However, that's slow and we don't expect inputs that are over 1024 tokens to happen that often anyway
        sorted_nlls = [s for s in sorted_nlls if s[1] != 0]

        # Misspell the words_to_alter most likely candidate words sorted by order of appearance
        edits = []
        for (i, s, e), nll in sorted(sorted_nlls[:words_to_alter], key=lambda x: x[0][0]):
            toks[i] = self.misspell(text[s:e])
            edits.append((len("".join(toks[:i])), len("".join(toks[: i + 1]))))

        return {"generation": "".join(toks), "num_edits": len(edits), "edits": edits}

    # From https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
    def calculate_perplexity(self, text):
        input_texts = [self.tokenizer.bos_token + text]
        inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(inputs["input_ids"])
        probs = torch.log_softmax(outputs.logits, dim=-1).detach().cpu()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = inputs["input_ids"][:, 1:].cpu()
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in self.tokenizer.all_special_ids:
                    text_sequence.append((self.tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        return batch[0]

    def misspell(self, w):
        misspellings = self.corrections.get(w.lower())
        choice = random.choice(misspellings)
        return choice.capitalize() if w[0].isupper() else choice

    def can_misspell(self, word):
        return word.lower() in self.corrections
