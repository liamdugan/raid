from tqdm import tqdm

from .gltr_model import LM


class GLTR:

    RANK = 10

    def __init__(self) -> None:
        self.model = LM()

    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            try:
                results = self.model.check_probabilities(text, topk=1)
            except IndexError as e:
                print(f"Error: {e}")
                predictions.append(1)  # Machine
                continue

            numbers = [item[0] for item in results["real_topk"]]
            numbers = [item <= self.RANK for item in numbers]

            predictions.append(sum(numbers) / len(numbers))

        return predictions

    @classmethod
    def set_param(cls, rank: int, prob: float) -> None:
        cls.RANK = rank
