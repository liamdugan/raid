import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Radar:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = AutoModelForSequenceClassification.from_pretrained(
            "TrustSafeAI/RADAR-Vicuna-7B", cache_dir=os.environ["CACHE_DIR"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
        self.detector.eval()
        self.detector.to(self.device)

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            with torch.no_grad():
                inputs = self.tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output_probs = F.log_softmax(self.detector(**inputs).logits, -1)[:, 0].exp().tolist()
            predictions.append(output_probs[0])
        return predictions
