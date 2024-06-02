import os

from tqdm import tqdm

from .fastdetectgpt.local_infer import FastDetectGPTModel


class FastDetectGPT:
    def __init__(self, use_log_rank=False):
        parent_dir = os.path.dirname(__file__)
        ref_path = os.path.join(parent_dir, "fastdetectgpt/local_infer_ref")
        self.fast_detect_gpt_instance = FastDetectGPTModel(
            scoring_model_name="gpt-neo-2.7B",
            reference_model_name="gpt-j-6B",
            cache_dir=os.environ["CACHE_DIR"],
            dataset="xsum",
            device="cuda",
            ref_path=ref_path,
            use_log_rank=use_log_rank,
        )

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            prob = self.fast_detect_gpt_instance.run(text)
            predictions.append(prob)
        return predictions

    def interactive(self):
        self.fast_detect_gpt_instance.run_interactive()
