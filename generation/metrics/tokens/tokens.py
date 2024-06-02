import os

import cohere
import numpy as np
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer

ERRORS = {
    cohere.error.CohereAPIError: "Cohere API Error:",
    cohere.error.CohereConnectionError: "Cohere Connection Error:",
    cohere.error.CohereError: "CohereError:",
}

model_name_map = {
    "chatgpt": "gpt-3.5-turbo-0613",
    "gpt3": "text-davinci-002",
    "gpt4": "gpt-4-0613",
    "cohere": "command",
    "cohere-chat": "command",
    "gpt2": "gpt2-xl",
    "llama": "meta-llama/Llama-2-70b-hf",
    "llama-chat": "meta-llama/Llama-2-70b-chat-hf",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-chat": "mistralai/Mistral-7B-Instruct-v0.1",
    "mpt": "mosaicml/mpt-30b",
    "mpt-chat": "mosaicml/mpt-30b-chat",
}


class Tokens:
    def __init__(self):
        # For each model, initialize the tokenizer
        # TODO: Move this to model.py (get tokenizer) or something
        self.tokenizers = dict()
        for k, v in model_name_map.items():
            if k in ["gpt2", "llama", "llama-chat", "mistral", "mistral-chat", "mpt", "mpt-chat"]:
                self.tokenizers[k] = AutoTokenizer.from_pretrained(v, trust_remote_code=("mpt" in v))
            elif k in ["chatgpt", "gpt4", "gpt3"]:
                self.tokenizers[k] = tiktoken.encoding_for_model(v)
            elif k in ["cohere", "cohere-chat"]:
                if os.environ["COHERE_API_KEY"] == "":
                    print("Warning: Cohere API key is not set. Add API key to api_keys.py and run the script.")
                    exit(-1)
                self.tokenizers[k] = cohere.Client(os.environ["COHERE_API_KEY"])

        # For each model, create anonymous tokenize functions using the tokenizers
        self.tokenize = dict()
        for k, v in model_name_map.items():
            if k in ["gpt2", "llama", "llama-chat", "mistral", "mistral-chat", "mpt", "mpt-chat"]:
                self.tokenize[k] = lambda x, y: self._huggingface_tokenize(self.tokenizers[y], x)
            elif k in ["chatgpt", "gpt4", "gpt3"]:
                self.tokenize[k] = lambda x, y: self._openai_tokenize(self.tokenizers[y], x)
            elif k in ["cohere", "cohere-chat"]:
                self.tokenize[k] = lambda x, y: self._cohere_tokenize(self.tokenizers[y], x)

    # Now we initialize the tokenize function for each model
    def _huggingface_tokenize(self, tokenizer, text):
        return len(tokenizer(text)["input_ids"])

    def _openai_tokenize(self, tokenizer, text):
        return len(tokenizer.encode(text))

    def _cohere_tokenize(self, tokenizer, text):
        response = tokenizer.tokenize(text=text)
        # If the request errored print the error and return nan
        if isinstance(response, tuple(ERRORS.keys())):
            print(response)
            return np.nan
        return len(response.tokens)

    def calculate_token_len(self, text, model):
        return self.tokenize[model](text, model)

    def compute(self, texts: list, models: list) -> list:
        metrics = []
        for text, model in tqdm(list(zip(texts, models))):
            # If text is human, use gpt4 tokenizer, otherwise use the model's tokenizer
            model = "gpt4" if model == "human" else model
            metrics.append(self.calculate_token_len(text, model))
        return metrics
