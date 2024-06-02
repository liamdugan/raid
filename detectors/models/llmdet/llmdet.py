import math

import datasets
import numpy as np
import transformers
from lightgbm import Booster
from tqdm import tqdm
from transformers import AutoTokenizer, BartTokenizer, LlamaTokenizer, T5Tokenizer
from unilm import UniLMTokenizer


class LLMDet:
    def __init__(self):
        # Load dictionary file from HuggingFace.
        dm = datasets.DownloadManager()
        files = dm.download_and_extract(
            "https://huggingface.co/datasets/TryMore/n_grams_probability/resolve/main/n-grams_probability.tar.gz"
        )
        model = ["gpt2", "opt", "unilm", "llama", "bart", "t5", "bloom", "neo", "vicuna", "gpt2_large", "opt_3b"]
        self.n_grams = dict()
        for item in model:
            n_grams = np.load(f"{files}/npz/{item}.npz", allow_pickle=True)
            self.n_grams[item] = n_grams["t5"]

        # Load classifier model
        model_files = dm.download_and_extract(
            "https://huggingface.co/datasets/TryMore/n_grams_probability/resolve/main/LightGBM_model.zip"
        )
        self.model = Booster(model_file=f"{model_files}/nine_LightGBM_model.txt")

        # List the names of all models we will use to calculate proxy probability
        self.model_information = [
            {"model_name": "gpt2", "vocab_size": 50265, "model_probability": "gpt2"},
            {"model_name": "facebook/opt-1.3b", "vocab_size": 50257, "model_probability": "opt"},
            {"model_name": "microsoft/unilm-base-cased", "vocab_size": 28996, "model_probability": "unilm"},
            {"model_name": "baffo32/decapoda-research-llama-7B-hf", "vocab_size": 32000, "model_probability": "llama"},
            {"model_name": "facebook/bart-base", "vocab_size": 50265, "model_probability": "bart"},
            {"model_name": "google/flan-t5-base", "vocab_size": 32128, "model_probability": "t5"},
            {"model_name": "bigscience/bloom-560m", "vocab_size": 250880, "model_probability": "bloom"},
            {"model_name": "EleutherAI/gpt-neo-2.7B", "vocab_size": 50257, "model_probability": "neo"},
            {"model_name": "lmsys/vicuna-7b-delta-v1.1", "vocab_size": 32000, "model_probability": "vicuna"},
            {"model_name": "gpt2-large", "vocab_size": 50265, "model_probability": "gpt2_large"},
            {"model_name": "facebook/opt-2.7b", "vocab_size": 50257, "model_probability": "opt_3b"},
        ]

        # Load tokenizers for each of the models in self.model_information
        transformers.logging.set_verbosity_error()
        transformers.logging.captureWarnings(True)
        self.tokenizers = dict()
        for model in self.model_information:
            if "unilm" in model["model_name"]:
                self.tokenizers[model["model_name"]] = UniLMTokenizer.from_pretrained(model["model_name"])
            elif "llama" in model["model_name"] or "vicuna" in model["model_name"]:
                self.tokenizers[model["model_name"]] = LlamaTokenizer.from_pretrained(model["model_name"], legacy=True)
            elif "t5" in model["model_name"]:
                self.tokenizers[model["model_name"]] = T5Tokenizer.from_pretrained(model["model_name"], legacy=True)
            elif "bart" in model["model_name"]:
                self.tokenizers[model["model_name"]] = BartTokenizer.from_pretrained(model["model_name"])
            else:
                self.tokenizers[model["model_name"]] = AutoTokenizer.from_pretrained(model["model_name"])
        transformers.logging.captureWarnings(False)
        transformers.logging.set_verbosity_warning()

    def perplexity(self, text_token_ids, n_grams_probability, vocab_size):
        """
        The `perplexity()` is used to calculate proxy perplexity with dictionary load in `load_probability()`.
        For each Language Model that has constructed an n-grams dictionary, a corresponding proxy perplexity will be computed."
        """
        ppl = 0
        number_3_grams = 0
        number_4_grams = 0
        number_2_grams = 0
        for i in range(2, len(text_token_ids) - 1):

            # Calculate the perplexity with 4-grams samples probability
            if tuple([text_token_ids[i - j] for j in range(2, -1, -1)]) in n_grams_probability[4].keys():
                if (
                    text_token_ids[i + 1]
                    in n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])]
                ):
                    if (
                        n_grams_probability[5][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])][
                            n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])]
                            .tolist()
                            .index(text_token_ids[i + 1])
                        ]
                        > 0
                    ):
                        ppl = ppl + math.log2(
                            n_grams_probability[5][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])][
                                n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])]
                                .tolist()
                                .index(text_token_ids[i + 1])
                            ]
                        )
                else:
                    top_k = len(n_grams_probability[4][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])])
                    sum_probs = sum(n_grams_probability[5][tuple([text_token_ids[i - j] for j in range(2, -1, -1)])])
                    if (1 - sum_probs) > 0:
                        ppl = ppl + math.log2((1 - sum_probs) / (vocab_size - top_k))
                number_4_grams = number_4_grams + 1

            # Calculate the perplexity with 3-grams samples probability
            elif tuple([text_token_ids[i - 1], text_token_ids[i]]) in n_grams_probability[2].keys():
                if text_token_ids[i + 1] in n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])]:
                    if (
                        n_grams_probability[3][tuple([text_token_ids[i - 1], text_token_ids[i]])][
                            n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])]
                            .tolist()
                            .index(text_token_ids[i + 1])
                        ]
                        > 0
                    ):
                        ppl = ppl + math.log2(
                            n_grams_probability[3][tuple([text_token_ids[i - 1], text_token_ids[i]])][
                                n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])]
                                .tolist()
                                .index(text_token_ids[i + 1])
                            ]
                        )
                else:
                    top_k = len(n_grams_probability[2][tuple([text_token_ids[i - 1], text_token_ids[i]])])
                    sum_probs = sum(n_grams_probability[3][tuple([text_token_ids[i - 1], text_token_ids[i]])])
                    if (1 - sum_probs) > 0:
                        ppl = ppl + math.log2((1 - sum_probs) / (vocab_size - top_k))
                number_3_grams = number_3_grams + 1

            # Calculate the perplexity with 2-grams samples probability
            elif tuple([text_token_ids[i]]) in n_grams_probability[0].keys():
                if text_token_ids[i + 1] in n_grams_probability[0][tuple([text_token_ids[i]])]:
                    if (
                        n_grams_probability[1][tuple([text_token_ids[i]])][
                            n_grams_probability[0][tuple([text_token_ids[i]])].tolist().index(text_token_ids[i + 1])
                        ]
                        > 0
                    ):
                        ppl = ppl + math.log2(
                            n_grams_probability[1][tuple([text_token_ids[i]])][
                                n_grams_probability[0][tuple([text_token_ids[i]])].tolist().index(text_token_ids[i + 1])
                            ]
                        )
                else:
                    top_k = len(n_grams_probability[0][tuple([text_token_ids[i]])])
                    sum_probs = sum(n_grams_probability[1][tuple([text_token_ids[i]])])
                    if (1 - sum_probs) > 0:
                        ppl = ppl + math.log2((1 - sum_probs) / (vocab_size - top_k))
                number_2_grams = number_2_grams + 1

        perplexity = ppl / (number_2_grams + number_3_grams + number_4_grams + 1)

        return -perplexity

    def inference(self, texts: list) -> list:
        with tqdm(total=len(self.model_information) * len(texts)) as pbar:
            perplexity_result = []
            for model in self.model_information:
                tokenizer = self.tokenizers[model["model_name"]]

                results = []
                for text in texts:
                    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                    perp = self.perplexity(token_ids, self.n_grams[model["model_probability"]], model["vocab_size"])
                    results.append(perp)
                    pbar.update(1)

                perplexity_result.append(results)

        # The input features of classifier
        features = np.stack([perplexity_result[i] for i in range(len(perplexity_result))], axis=1)

        # Load classifier model
        y_pred = self.model.predict(features)
        label = ["Human_write", "GPT-2", "OPT", "UniLM", "LLaMA", "BART", "T5", "Bloom", "GPT-neo"]
        result = [{label[i]: y_pred[j][i] for i in range(len(label))} for j in range(len(y_pred))]

        # Final prediction is 1 - Pr(Human)
        for i in range(len(result)):
            result[i] = 1 - result[i]["Human_write"]

        return result
