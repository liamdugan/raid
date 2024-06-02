from .models.binoculars.binoculars import Binoculars
from .models.chatgpt_roberta_detector.chatgpt_detector import ChatGPTDetector
from .models.detectgpt.detectgpt import DetectGPT
from .models.fast_detectgpt.fast_detectgpt import FastDetectGPT
from .models.gltr.gltr import GLTR
from .models.gpt2_detector.gpt2_detector import GPT2Detector
from .models.gptzero.gptzero import GPTZero
from .models.llmdet.llmdet import LLMDet
from .models.originality_ai.originality_ai import OriginalityAI
from .models.radar.radar import Radar
from .models.winston_ai.winston_ai import WinstonAI
from .models.zerogpt.zerogpt import ZeroGPT


class Detector:
    """Shared interface for all detectors"""

    def inference(self, texts: list) -> list:
        """Takes in a list of texts and outputs a list of scores from 0 to 1 with
        0 indicating likely human-written, and 1 indicating likely machine-generated."""
        pass


def get_detector(detector_name: str) -> Detector:
    if detector_name == "gpt2-base":
        return GPT2Detector("gpt2-base")
    elif detector_name == "gpt2-large":
        return GPT2Detector("gpt2-large")
    elif detector_name == "gltr":
        return GLTR()
    elif detector_name == "chatgpt-roberta":
        return ChatGPTDetector()
    elif detector_name == "gptzero":
        return GPTZero()
    elif detector_name == "zerogpt":
        return ZeroGPT()
    elif detector_name == "detectgpt":
        return DetectGPT()
    elif detector_name == "fastdetectgpt":
        return FastDetectGPT()
    elif detector_name == "fastdetectllm":
        return FastDetectGPT(use_log_rank=True)
    elif detector_name == "llmdet":
        return LLMDet()
    elif detector_name == "radar":
        return Radar()
    elif detector_name == "winston_ai":
        return WinstonAI()
    elif detector_name == "originality_ai":
        return OriginalityAI()
    elif detector_name == "binoculars":
        return Binoculars()
    else:
        raise ValueError("Invalid detector name")
