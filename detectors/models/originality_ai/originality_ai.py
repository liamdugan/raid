import json
import os

import requests
from tqdm import tqdm


class OriginalityAI:
    def __init__(self):
        self.api_key = os.environ["ORIGINALITY_API_KEY"]
        if self.api_key == "":
            print("Warning: OriginalityAI API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

    def inference(self, texts: list) -> list:
        url = "https://api.originality.ai/api/v1/scan/ai"
        headers = {"X-OAI-API-KEY": self.api_key, "Accept": "application/json", "Content-Type": "application/json"}
        predictions = []
        for i, text in enumerate(tqdm(texts)):
            payload = json.dumps({"content": text, "aiModelVersion": "1", "storeScan": '"false"'})
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code == 200:
                response_json = response.json()
                ai_score = response_json.get("score", {}).get("ai", -1)  # -1 as a placeholder for error
                predictions.append(ai_score)  # Convert to percentage
            else:
                print(f"OriginalityAI returned a status code {response.status_code} error: {response}\n")
                predictions.append(-1)  # Using -1 as a placeholder for error
        return predictions
