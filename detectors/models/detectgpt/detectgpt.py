import functools
import os
import re

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


class DetectGPTModel:
    def __init__(
        self,
        pct_words_masked=0.3,
        span_length=2,
        n_perturbation_list="1,10",
        n_perturbation_rounds=1,
        base_model_name="gpt2-medium",
        scoring_model_name="",
        mask_filling_model_name="t5-large",
        batch_size=1,
        chunk_size=1,
        n_similarity_samples=20,
        int8=False,
        half=False,
        base_half=False,
        do_top_k=False,
        top_k=40,
        do_top_p=False,
        top_p=0.96,
        output_name="",
        openai_model=None,
        openai_key=None,
        baselines_only=False,
        buffer_size=1,
        mask_top_p=1.0,
        pre_perturb_pct=0.0,
        pre_perturb_span_length=5,
        random_fills=False,
        random_fills_tokens=False,
        cache_dir="~/.cache",
        min_words=40,
    ):
        self.pct_words_masked = pct_words_masked
        self.span_length = span_length
        self.n_perturbation_list = n_perturbation_list
        self.n_perturbation_rounds = n_perturbation_rounds
        self.base_model_name = base_model_name
        self.scoring_model_name = scoring_model_name
        self.mask_filling_model_name = mask_filling_model_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.n_similarity_samples = n_similarity_samples
        self.int8 = int8
        self.half = half
        self.base_half = base_half
        self.do_top_k = do_top_k
        self.top_k = top_k
        self.do_top_p = do_top_p
        self.top_p = top_p
        self.output_name = output_name
        self.openai_model = openai_model
        self.openai_key = openai_key
        self.baselines_only = baselines_only
        self.buffer_size = buffer_size
        self.mask_top_p = mask_top_p
        self.pre_perturb_pct = pre_perturb_pct
        self.pre_perturb_span_length = pre_perturb_span_length
        self.random_fills = random_fills
        self.random_fills_tokens = random_fills_tokens
        self.cache_dir = cache_dir
        self.device = "cuda"
        self.min_words = min_words
        self.base_model_name = base_model_name

        self.mask_tokenizer = None
        self.base_tokenizer = None
        self.base_model = None
        self.fill_dictionary = None
        self.mask_model = None
        self.gpt2_tokenizer = None

        self.n_perturbation_list = [int(x) for x in self.n_perturbation_list.split(",")]
        self.gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir=self.cache_dir)

        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_name, cache_dir=self.cache_dir
        ).to(self.device)
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model_name, cache_dir=self.cache_dir)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        # mask filling t5 model
        self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.mask_filling_model_name, cache_dir=self.cache_dir
        ).to(self.device)
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.mask_filling_model_name,
            truncation=True,
            cache_dir=self.cache_dir,
            model_max_length=self.mask_model.config.n_positions,
        )

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):
        tokens = text.split(" ")
        mask_string = "<<<mask>>>"

        n_spans = pct * len(tokens) / (span_length + self.buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = " ".join(tokens)
        return text

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.mask_model.generate(
            **tokens,
            max_length=150,
            do_sample=True,
            top_p=self.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(" ") for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def perturb_texts_(self, texts, span_length, pct, ceil_pct=False):
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
        return perturbed_texts

    def perturb_texts(self, texts, span_length, pct, ceil_pct=False):
        chunk_size = self.chunk_size
        if "11b" in self.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        for i in range(0, len(texts), chunk_size):
            outputs.extend(self.perturb_texts_(texts[i : i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
        return outputs

    def drop_last_word(self, text):
        return " ".join(text.split(" ")[:-1])

    # Get the log likelihood of each text under the base_model
    def get_ll(self, text):
        if len(text) == 0:
            return 0.0
        with torch.no_grad():
            tokenized = self.base_tokenizer(text, return_tensors="pt").to(self.device)
            labels = tokenized.input_ids
            return -self.base_model(**tokenized, labels=labels).loss.item()

    def get_lls(self, texts):
        return [self.get_ll(text) for text in texts]

    def get_perturbation_results(self, data, n_perturbations):
        torch.manual_seed(0)
        np.random.seed(0)

        results = []
        original_text = data["original"]

        perturb_fn = functools.partial(self.perturb_texts, span_length=self.span_length, pct=self.pct_words_masked)

        p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
        for _ in range(self.n_perturbation_rounds - 1):
            try:
                p_original_text = perturb_fn(p_original_text)
            except AssertionError:
                break

        assert (
            len(p_original_text) == len(original_text) * n_perturbations
        ), f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

        for idx in range(len(original_text)):
            results.append(
                {
                    "original": original_text[idx],
                    "perturbed_original": p_original_text[idx * n_perturbations : (idx + 1) * n_perturbations],
                }
            )

        for res in results:
            p_original_ll = self.get_lls(res["perturbed_original"])
            res["original_ll"] = self.get_ll(res["original"])
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

        return results

    def run_perturbation_experiment(self, results, criterion, n_perturbations):
        # compute diffs with perturbed
        predictions = {"real": []}
        for res in results:
            if criterion == "d":
                predictions["real"].append(res["original_ll"] - res["perturbed_original_ll"])
            elif criterion == "z":
                if res["perturbed_original_ll_std"] == 0:
                    res["perturbed_original_ll_std"] = 1
                predictions["real"].append(
                    (res["original_ll"] - res["perturbed_original_ll"]) / res["perturbed_original_ll_std"]
                )

        name = f"perturbation_{n_perturbations}_{criterion}"
        return {
            "name": name,
            "predictions": predictions,
            "info": {
                "pct_words_masked": self.pct_words_masked,
                "span_length": self.span_length,
                "n_perturbations": n_perturbations,
            },
            "raw_results": results,
        }

    # strip newlines from each example; replace one or more newlines with a single space
    def strip_newlines(self, text):
        return " ".join(text.split())

    # trim to shorter length
    def trim_to_shorter_length(self, texta, textb):
        # truncate to shorter of o and s
        shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
        texta = " ".join(texta.split(" ")[:shorter_length])
        textb = " ".join(textb.split(" ")[:shorter_length])
        return texta, textb

    def truncate_to_substring(text, substring, idx_occurrence):
        # truncate everything after the idx_occurrence occurrence of substring
        assert idx_occurrence > 0, "idx_occurrence must be > 0"
        idx = -1
        for _ in range(idx_occurrence):
            idx = text.find(substring, idx + 1)
            if idx == -1:
                return text
        return text[:idx]

    def detect(self, text):
        text = self.strip_newlines(text.strip())
        # truncate text to max length of tokenizer (todo optimize this somehow)
        text = self.mask_tokenizer.decode(self.mask_tokenizer(text)["input_ids"][:-1][:512])
        data = pd.DataFrame({"original": [text]})

        outputs = []

        # run perturbation experiments
        for n_perturbations in self.n_perturbation_list:
            perturbation_results = self.get_perturbation_results(data, n_perturbations)
            for perturbation_mode in ["z"]:
                output = self.run_perturbation_experiment(perturbation_results, perturbation_mode, n_perturbations)
                outputs.append(output)

        return outputs


class DetectGPT:
    def __init__(self):
        self.detect_gpt_instance = DetectGPTModel(
            pct_words_masked=0.15, n_perturbation_list="10", cache_dir=os.environ["CACHE_DIR"]
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm.tqdm(texts):
            initial = self.detect_gpt_instance.detect(text)
            out = initial[0].get("predictions")
            predictions.append(out.get("real")[0])

        return predictions
