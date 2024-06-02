import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Perplexity:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.environ["CACHE_DIR"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Get the location in the model config where max sequence length is
        if "max_position_embeddings" in self.model.config.attribute_map:
            key = self.model.config.attribute_map["max_position_embeddings"]
        else:
            key = "max_position_embeddings"

        try:
            self.max_length = self.model.config.to_dict()[key]
        except KeyError:
            # If we can't find the max length then error
            raise KeyError("Error: Could not find max sequence length")

        self.stride = self.max_length // 2

    def calculate_perplexity(self, text):
        """
        Source: https://huggingface.co/docs/transformers/perplexity
        """
        encoding = self.tokenizer(text, return_tensors="pt")
        seq_len = encoding.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encoding.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        return torch.exp(torch.stack(nlls).mean()).cpu().item()

    def compute(self, texts: list) -> list:
        metrics = []
        for text in tqdm(texts):
            metrics.append(self.calculate_perplexity(text))
        return metrics
