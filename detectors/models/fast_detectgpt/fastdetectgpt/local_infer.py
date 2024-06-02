# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import os

import numpy as np
import torch

from .fast_detect_gpt import get_log_sampling_discrepancy_analytic, get_sampling_discrepancy_analytic
from .model import load_model, load_tokenizer


# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, ref_path, use_log_rank):
        self.use_log_rank = use_log_rank
        self.ref_path = ref_path
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(self.ref_path, "*.json")):
            with open(result_file, "r") as fin:
                res = json.load(fin)
                self.real_crits.extend(res["predictions"]["real"])
                self.fake_crits.extend(res["predictions"]["samples"])
        print(f"ProbEstimator: total {len(self.real_crits) * 2} samples.")

    def crit_to_prob(self, crit):
        # Do not perform probability estimation when using log-rank
        if self.use_log_rank:
            return crit

        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)


class FastDetectGPTModel:
    def __init__(self, scoring_model_name, reference_model_name, cache_dir, dataset, device, ref_path, use_log_rank):
        self.device = device
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.scoring_model_name = scoring_model_name
        self.reference_model_name = reference_model_name
        self.ref_path = ref_path

        # load model
        self.scoring_tokenizer = load_tokenizer(self.scoring_model_name, self.dataset, self.cache_dir)
        self.scoring_model = load_model(self.scoring_model_name, self.device, self.cache_dir)
        self.scoring_model.eval()
        if self.reference_model_name != self.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.reference_model_name, self.dataset, self.cache_dir)
            self.reference_model = load_model(self.reference_model_name, self.device, self.cache_dir)
            self.reference_model.eval()

        # evaluate criterion
        self.name = "sampling_discrepancy_analytic"
        if use_log_rank:
            self.criterion_fn = get_log_sampling_discrepancy_analytic
        else:
            self.criterion_fn = get_sampling_discrepancy_analytic

        self.prob_estimator = ProbEstimator(self.ref_path, use_log_rank)

    def run(self, text):
        tokenized = self.scoring_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False
        ).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.reference_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False
                ).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated text
        prob = self.prob_estimator.crit_to_prob(crit)
        # print(f'Fast-DetectGPT criterion is {crit:.4f}')
        return prob

    def run_interactive(self):
        # evaluate text
        print("Local demo for Fast-DetectGPT, where the longer text has more reliable result.")
        print("")
        while True:
            print("Please enter your text: (Press Enter twice to start processing)")
            lines = []
            while True:
                line = input()
                if len(line) == 0:
                    break
                lines.append(line)
            text = "\n".join(lines)
            if len(text) == 0:
                break
            self.run(text)
