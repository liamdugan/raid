import os
from glob import glob

import torch
import wget
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class GPT2Detector:

    path_to_weights = os.path.abspath("detectors/gpt2_detector/openai-gpt-2-detector/detector_weights")
    if not os.path.exists(path_to_weights):
        os.makedirs(path_to_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model) -> None:
        self.model_name = model
        self._check_weights()
        self._init_model()

    def _check_weights(self):
        """
        Check if the weights for the given model are present in the current directory.
        If not, download them from the repository.
        """
        if self.model_name == "gpt2-base":
            weights_name = "detector-base.pt"
            if not glob(os.path.join(self.path_to_weights, weights_name)):
                print("Downloading weights for GPT2 Base Detector...")
                wget.download(
                    "https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt",
                    out=os.path.join(self.path_to_weights, weights_name),
                )

        elif self.model_name == "gpt2-large":
            weights_name = "detector-large.pt"
            if not glob(os.path.join(self.path_to_weights, weights_name)):
                print("Downloading weights for GPT2 Large Detector...")
                wget.download(
                    "https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt",
                    out=os.path.join(self.path_to_weights, weights_name),
                )

    def _init_model(self):
        model_name = "roberta-large" if self.model_name == "gpt2-large" else "roberta-base"
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        if self.model_name == "gpt2-base":
            weight_name = "detector-base.pt"
        else:
            weight_name = "detector-large.pt"

        checkpoint = os.path.join(self.path_to_weights, weight_name)
        data = torch.load(checkpoint, map_location="cpu")

        self.model.load_state_dict(data["model_state_dict"], strict=False)
        self.model.eval()

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            tokens = self.tokenizer.encode(text)
            tokens = tokens[: self.tokenizer.model_max_length - 2]
            tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]).unsqueeze(0)
            mask = torch.ones_like(tokens)

            self.model = self.model.to(self.device)
            with torch.no_grad():
                logits = self.model(tokens.to(self.device), attention_mask=mask.to(self.device))[0]
                probs = logits.softmax(dim=-1)

            fake, real = probs.detach().cpu().flatten().numpy().tolist()
            predictions.append(fake)
        return predictions
