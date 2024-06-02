import os
import time

import requests
from tqdm import tqdm


class ZeroGPT:
    def __init__(self):
        self.api_key = os.environ["ZEROGPT_API_KEY"]
        if self.api_key == "":
            print("Warning: ZeroGPT API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

        self.url = "https://api.zerogpt.com/api/detect/detectText"
        self.headers = {"ApiKey": self.api_key}

    def inference(self, texts: list) -> list:
        """
        Run ZeroGPT on the given text.
        """
        predictions = []
        for i, text in enumerate(tqdm(texts)):
            res = requests.post(self.url, headers=self.headers, json={"input_text": text})
            if res.status_code == 200:
                results = res.json()
                if results["code"] == 200:  # ???
                    if results["data"] and ("isHuman" in results["data"]):
                        score = results["data"].get("isHuman") / 100
                        predictions.append(1 - score)
                    else:
                        predictions.append(-1)
                else:
                    predictions.append(-1)
            else:
                print(f"ZeroGPT returned a status code {res.status_code} error: {res.content}\n")
                predictions.append(-1)
                if res.status_code == 429:
                    print("Got rate limit error - sleeping for 60 seconds...")
                    time.sleep(60)
        return predictions
