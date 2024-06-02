import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ChatGPTDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
        self.model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta").to(
            self.device
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            inputs = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            real, fake = probs.detach().cpu().flatten().numpy().tolist()
            predictions.append(fake)
        return predictions
