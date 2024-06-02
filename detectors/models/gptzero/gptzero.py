import os
import time

import requests
from tqdm import tqdm


class GPTZero:
    def __init__(self):
        self.api_key = os.environ["GPTZERO_API_KEY"]
        if self.api_key == "":
            print("Warning: GPTZero API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

    def inference(self, texts: list) -> list:
        """
        Run GPTZero on the given text.
        """
        url = "https://api.gptzero.me/v2/predict/text"
        headers = {"x-api-key": self.api_key}
        predictions = []
        for i, text in enumerate(tqdm(texts)):
            payload = {"document": text}
            res = requests.post(url, headers=headers, json=payload)
            if res.status_code == 200:
                results = res.json()
                predictions.append(results["documents"][0]["completely_generated_prob"])
            else:
                print(f"GPTZero returned a status code {res.status_code} error: {res}\n")
                predictions.append(-1)
                if res.status_code == 429:
                    print("Got rate limit error - sleeping for 60 seconds...")
                    time.sleep(60)
        return predictions
