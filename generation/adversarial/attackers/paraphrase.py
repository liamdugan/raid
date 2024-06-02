import itertools
import os

import nltk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ParaphraseAttack:
    def __init__(
        self, N=1.0, sent_interval=3, include_prefix=True, preserve_separators=False, do_split_on_newlines=False
    ):
        # The number of sentences we paraphrase simultaneously
        self.sent_interval = sent_interval
        # Whether or not to include the previously paraphrased sentences as context to the model
        self.include_prefix = include_prefix
        # Whether or not to preserve the correct separating whitespace tokens between each set of sent_interval sentences
        self.preserve_separators = preserve_separators
        # Whether or not to split sentences on newlines in addition to normal sentence tokenization
        self.do_split_on_newlines = do_split_on_newlines

        # Use no context model if not including prefix
        self.model_name = "kalpeshk2011/dipper-paraphraser-xxl"
        if not self.include_prefix:
            self.model_name += "-no-context"

        # Load in the model at half precision
        self.tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", legacy=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, cache_dir=os.environ["CACHE_DIR"], device_map="sequential", torch_dtype=torch.float16
        )

    def split_on_newlines(self, text, all_spans):
        new_spans = []
        for index, span in enumerate(all_spans):
            # Extract the text of the span
            snippet = text[span[0] : span[1]]

            # If we are a separator or the sentence doesn't have a newline, add it
            if index % 2 == 1 or "\n" not in snippet:
                new_spans.append(span)
                continue

            # Otherwise, split on newline
            splits = snippet.split("\n")

            # Add a span for each split and the newline after it
            spans = []
            i = span[0]
            for s in splits:
                spans.append((i, i + len(s)))
                spans.append((i + len(s), i + len(s) + 1))
                i = i + len(s) + 1
            spans = spans[:-1]

            # Insert these new spans into the original list
            new_spans.extend(spans)

        return new_spans

    def truncate_text(self, text, length):
        input_ids = self.tokenizer(text)["input_ids"]
        amount_truncated = len(input_ids) - len(input_ids[length:])
        return self.tokenizer.decode(input_ids[length:]), amount_truncated

    def get_input_text(self, lex_code, order_code, output_text, curr_sent_window):
        prefix = f"lexical = {lex_code}, order = {order_code} "
        suffix = f"{output_text} <sent> {curr_sent_window} </sent>" if self.include_prefix else curr_sent_window

        # Get the length in tokens for the prefix and suffix and calculate surplus
        len_prefix = len(self.tokenizer(prefix)["input_ids"])
        len_suffix = len(self.tokenizer(suffix)["input_ids"])
        surplus = (len_suffix + len_prefix) - self.tokenizer.model_max_length

        # If we are over the max length and have context, truncate context
        if surplus > 0 and self.include_prefix:
            output_text, amount_truncated = self.truncate_text(output_text, surplus)
            surplus -= amount_truncated

        # If we are still over max len, truncate the current sentences
        if surplus > 0:
            curr_sent_window, amount_truncated = self.truncate_text(curr_sent_window, surplus)
            surplus -= amount_truncated

        suffix = f"{output_text} <sent> {curr_sent_window} </sent>" if self.include_prefix else curr_sent_window

        return prefix + suffix

    def attack(self, text, lex_diversity=60, order_diversity=0):
        """
        This function generates a paraphrased version of the input text.

        Args:
            text (str): The text to be paraphrased.
            lex_diversity (int, optional): Lexical diversity of the output. Defaults to 60.
            order_diversity (int, optional): Order diversity of the output. Defaults to 0.

        Returns:
            str: The paraphrased text.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sent_spans = sent_tokenizer.span_tokenize(text)
        all_spans = list(itertools.pairwise(itertools.chain.from_iterable(sent_spans)))

        if self.do_split_on_newlines:
            all_spans = self.split_on_newlines(text, all_spans)

        sent_tokens = [text[s:e] for s, e in all_spans]

        # If not preserving separators, convert all whitespace to spaces (match original code)
        if not self.preserve_separators:
            sentences = nltk.sent_tokenize(" ".join(text.split()))
            spaces = [" "] * len(sentences)
            sent_tokens = list(itertools.chain(*zip(sentences, spaces)))[:-1]

        output_text = ""
        for i in range(0, len(sent_tokens), self.sent_interval * 2):
            # Get the current window of sent tokens
            tokens = sent_tokens[i : i + (self.sent_interval * 2)]

            # If len tokens is even, then remove the final token and append it on after paraphrasing
            separator = tokens[-1] if len(tokens) % 2 == 0 else ""
            curr_sent_window = "".join(tokens[:-1]) if len(tokens) % 2 == 0 else "".join(tokens)

            # Get the input text to the paraphrase model
            input_text = self.get_input_text(lex_code, order_code, output_text, curr_sent_window)

            # Pass text through the paraphrasing model
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, do_sample=True, top_p=0.75, top_k=None, max_length=512)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Append text to the output
            output_text += decoded[0] + separator

        return {"generation": output_text.strip(), "num_edits": 1, "edits": [(0, len(output_text))]}
