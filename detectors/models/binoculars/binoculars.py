import os

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.metrics import entropy, perplexity
from .utils.utils import assert_tokenizer_consistency

torch.set_grad_enabled(False)

GLOBAL_BINOCULARS_THRESHOLD = 0.9015310749276843  # selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(object):
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
    ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            cache_dir=os.environ["CACHE_DIR"],
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            cache_dir=os.environ["CACHE_DIR"],
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )

        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text):
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text):
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < GLOBAL_BINOCULARS_THRESHOLD, "AI-Generated", "Human-Generated").tolist()
        return pred

    def inference(self, texts: list[str]):
        """Note:
            Returns `-score` for compatibility with greater-than-thresholding in downstream evaluation,
            as the Binoculars score is *not* bound to the domain [0, 1].
        """
        predictions = []
        for text in tqdm(texts):
            predictions.append(-self.compute_score(text))
        return predictions
