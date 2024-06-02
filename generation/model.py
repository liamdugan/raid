from models.cohere import CohereGenerator
from models.huggingface import HuggingGenerator
from models.openai import OpenAIGenerator


class Model:
    """Shared interface for all generators"""

    def generate(self, prompts: list, config: dict) -> list:
        """Takes in a list of prompts and outputs a list of generations"""
        pass

    def get_config(self, decoding: str, add_penalty: str) -> dict:
        """Takes in a decoding strategy and penalty and outputs a configuration dict"""
        pass


def get_models_list():
    """List of all legal generative models"""
    return [
        "chatgpt",
        "gpt3",
        "gpt4",
        "cohere",
        "cohere-chat",
        "gpt2",
        "llama",
        "llama-chat",
        "mistral",
        "mistral-chat",
        "mpt",
        "mpt-chat",
    ]


def is_chat_model(model: str) -> bool:
    """Given a model name, return if it's a chat model"""
    return model in ["chatgpt", "gpt4", "llama-chat", "mistral-chat", "cohere-chat", "mpt-chat"]


def is_api_model(model: str) -> bool:
    """Given a model name, return if it's an API model"""
    return model in ["chatgpt", "gpt4", "gpt3", "cohere", "cohere-chat"]


def get_model(name: str) -> Model:
    """Given a model name, return the Model object"""
    if name == "chatgpt":
        return OpenAIGenerator("gpt-3.5-turbo-0613", is_chat_model=True)
    elif name == "gpt3":
        return OpenAIGenerator("text-davinci-002", is_chat_model=False)
    elif name == "gpt4":
        return OpenAIGenerator("gpt-4-0613", is_chat_model=True)
    elif name == "cohere":
        return CohereGenerator("command")
    elif name == "cohere-chat":
        return CohereGenerator("command", is_chat_model=True)
    elif name == "gpt2":
        return HuggingGenerator("gpt2-xl")
    elif name == "llama":
        return HuggingGenerator("meta-llama/Llama-2-70b-hf")
    elif name == "llama-chat":
        return HuggingGenerator("meta-llama/Llama-2-70b-chat-hf", is_chat_model=True)
    elif name == "mistral":
        return HuggingGenerator("mistralai/Mistral-7B-v0.1")
    elif name == "mistral-chat":
        return HuggingGenerator("mistralai/Mistral-7B-Instruct-v0.1", is_chat_model=True)
    elif name == "mpt":
        return HuggingGenerator("mosaicml/mpt-30b")
    elif name == "mpt-chat":
        return HuggingGenerator("mosaicml/mpt-30b-chat", is_chat_model=True)
    else:
        raise ValueError("Invalid model name")
