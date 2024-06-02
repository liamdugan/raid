from .perplexity.perplexity import Perplexity
from .selfbleu.selfbleu import SelfBLEU
from .tokens.tokens import Tokens


class Metric:
    """Shared interface for all metric calculations"""

    def compute(self, texts: list) -> list:
        """Takes in a list of texts and outputs a list containing the score from the metric"""
        pass


def get_metric(metric_name: str) -> Metric:
    if metric_name == "perplexity_gpt2":
        return Perplexity("gpt2")
    elif metric_name == "perplexity_gpt2xl":
        return Perplexity("gpt2-xl")
    elif metric_name == "perplexity_llama":
        return Perplexity("meta-llama/Llama-2-7b-hf")
    elif metric_name == "selfbleu":
        return SelfBLEU()
    elif metric_name == "tokens":
        return Tokens()
    else:
        raise ValueError("Invalid metric name")
