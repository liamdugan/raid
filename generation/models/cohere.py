import asyncio
import os

import cohere

ERRORS = {
    cohere.error.CohereAPIError: "Cohere API Error:",
    cohere.error.CohereConnectionError: "Cohere Connection Error:",
    cohere.error.CohereError: "CohereError:",
}


class CohereGenerator:
    def __init__(self, model_name, is_chat_model=False):
        self.model = model_name
        self.batch_size = 20
        self.verbose = True
        self.cohere_key = os.environ["COHERE_API_KEY"]
        self.is_chat_model = is_chat_model

        if self.cohere_key == "":
            print("Warning: Cohere API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

    async def _async_generate(self, texts: list, config: dict) -> list:
        async with cohere.AsyncClient(self.cohere_key) as aio_co:
            generations = [""] * len(texts)
            for i in range(0, len(texts), self.batch_size):
                # Grab the batch of texts
                batch = texts[i : i + self.batch_size]

                # Create the set of requests
                if self.is_chat_model:
                    requests = [aio_co.chat(message=x, model=self.model, **config) for x in batch]
                else:
                    requests = [aio_co.generate(prompt=x, model=self.model, **config) for x in batch]

                # Wait on the responses
                responses = await asyncio.gather(*requests, return_exceptions=True)

                # Add text of successful queries to generations
                for j, response in enumerate(responses):
                    if not isinstance(response, tuple(ERRORS.keys())):
                        generations[i + j] = response.text if self.is_chat_model else response[0].text
                    else:
                        if self.verbose:
                            print(ERRORS[type(response)], response)

        return generations

    def generate(self, prompts: list, config: dict) -> list:
        return asyncio.run(self._async_generate(prompts, config))

    def get_config(self, decoding: str, add_penalty: str = "no") -> dict:
        if decoding not in ["greedy", "sampling"]:
            return None

        config = {
            "temperature": 1 if decoding == "sampling" else 0,
            "p": 1 if decoding == "sampling" else 0,
            "max_tokens": 512,
        }

        if add_penalty == "yes":
            # Using frequency penalty of roughly 0.5 as was done in https://arxiv.org/pdf/2311.01873.pdf
            config["frequency_penalty"] = 0.5

        return config
