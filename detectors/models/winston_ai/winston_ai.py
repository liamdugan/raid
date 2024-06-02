import json
import os
import time

import requests
from tqdm import tqdm


class WinstonAI:
    def __init__(self):
        self.api_key = os.environ["WINSTON_API_KEY"]

        if self.api_key == "":
            print("Warning: WinstonAI API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

        with open(f"{os.path.dirname(__file__)}/config.json", "r") as file:
            config = json.load(file)
            self.headers = config["headers"]

    def inference(self, texts: list) -> list:
        """
        Run WinstonAI on the given texts.

        :param texts: The texts to evaluate
        :return: The result of the API call as a list of integers:
                 0 if human-written, 1 if machine-generated, and -1 if there was an error
        """
        url = "https://api.gowinston.ai/functions/v1/predict"
        predictions = []
        for i, text in enumerate(tqdm(texts)):

            # Winston does not support text under 300 characters
            if len(text) < 300:
                predictions.append(-1)
                continue

            payload = json.dumps(
                {"api_key": self.api_key, "text": text, "sentences": False, "language": "en", "version": "3.0"}
            )

            try:
                response = requests.request("POST", url, headers=self.headers, data=payload)
                if response.status_code == 200:
                    response_json = response.json()
                    if "score" in response_json:
                        score = response_json.get("score") / 100
                        predictions.append(1 - score)
                    else:
                        predictions.append(-1)
                else:
                    print(f"WinstonAI returned a status code {response.status_code} error:\n", response.text)
                    predictions.append(-1)
                    if response.status_code == 429:
                        print("Got rate limit error - sleeping for 60 seconds...")
                        time.sleep(60)
            except (ConnectionError, ConnectionResetError, requests.exceptions.ConnectionError) as e:
                print(f"Error: {e} - sleeping for 2 minutes...")
                time.sleep(120)
        return predictions
