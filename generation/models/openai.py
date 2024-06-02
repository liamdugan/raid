import asyncio
import os

import openai
from aiohttp import ClientSession

ERRORS = {
    openai.error.InvalidRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    openai.error.RateLimitError: "OpenAI API rate limit exceeded",
    openai.error.APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",
    openai.error.Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    openai.error.ServiceUnavailableError: "OpenAI service unavailable error",
    openai.error.APIError: "OpenAI API error",
}


class OpenAIGenerator:
    def __init__(self, model_name, is_chat_model=True):
        self.model = model_name
        self.batch_size = 20
        self.verbose = False
        self.api_key = os.environ["OPENAI_API_KEY"]

        if self.api_key == "":
            print("Warning: OpenAI API key is not set. Add API key to api_keys.py and run the script.")
            exit(-1)

        if is_chat_model:
            self._generate = lambda x, config: openai.ChatCompletion.acreate(model=self.model, messages=x, **config)
            self._get_response = lambda x: x["choices"][0]["message"]["content"]
            self._build_prompt = lambda x: [{"role": "assistant", "content": x}]
        else:
            self._generate = lambda x, config: openai.Completion.acreate(model=self.model, prompt=x, **config)
            self._get_response = lambda x: x["choices"][0]["text"]
            self._build_prompt = lambda x: x

    def _set_api_key(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        openai.organization = os.environ["OPENAI_ORG_ID"]

    async def _async_generate(self, texts: list, config: dict) -> list:
        openai.aiosession.set(ClientSession())
        generations = [""] * len(texts)
        for i in range(0, len(texts), self.batch_size):
            # Grab the batch of texts
            batch = texts[i : i + self.batch_size]

            # Concatenate prompts to all elements and put in message format
            messages = [self._build_prompt(b) for b in batch]

            # Create asyncio requests for the batch
            requests = [self._generate(x, config) for x in messages]

            # Dispatch the requests and wait on their completion
            responses = await asyncio.gather(*requests, return_exceptions=True)

            # For all successful queries, mark them as complete and update progress bar
            for j, response in enumerate(responses):
                if not isinstance(response, tuple(ERRORS.keys())):
                    generations[i + j] = self._get_response(response)
                else:
                    if self.verbose:
                        print(ERRORS[type(response)], response)

            # If we got any rate limit errors wait 10s before next batch
            if any([isinstance(r, openai.error.RateLimitError) for r in responses]):
                if self.verbose:
                    print("Sleeping for 10 seconds due to RateLimitError.")
                await asyncio.sleep(10)

        await openai.aiosession.get().close()
        return generations

    def generate(self, prompts: list, config: dict) -> list:
        self._set_api_key()
        return asyncio.run(self._async_generate(prompts, config))

    def get_config(self, decoding: str, add_penalty: str = "no") -> dict:
        if decoding not in ["greedy", "sampling"]:
            return None

        config = {
            "temperature": 1 if decoding == "sampling" else 0,
            "top_p": 1 if decoding == "sampling" else 0,
            "max_tokens": 512,
        }

        if add_penalty == "yes":
            # Using frequency penalty of roughly 0.5 as was done in https://arxiv.org/pdf/2311.01873.pdf
            config["frequency_penalty"] = 0.5

        return config
