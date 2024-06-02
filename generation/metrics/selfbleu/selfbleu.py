import numpy as np
from nltk.tokenize import sent_tokenize
from sacrebleu.metrics import BLEU
from tqdm import tqdm


class SelfBLEU:
    def __init__(self):
        self.bleu = BLEU(trg_lang="en")

    def calculate_selfbleu(self, text):
        # Start by sentence tokenizing the text.
        sents = sent_tokenize(text)

        # If we have 0 or 1 sentences, we cannot calculate self-BLEU
        if len(sents) < 2:
            return np.nan

        # Initialize the sys and refs arrays
        sys = []
        refs = []

        # Loop such that all sentences have the other sentences
        # in the passage as their reference sentences
        for i in range(len(sents) - 1):
            sys.append(sents[i])
            refs.append(sents[i + 1 :] + sents[: i + 1])

        # Add the last sentence on to sys
        sys.append(sents[-1])

        # Return the self-BLEU score
        return self.bleu.corpus_score(sys, refs).score

    def compute(self, texts: list) -> list:
        metrics = []
        for text in tqdm(texts):
            metrics.append(self.calculate_selfbleu(text))
        return metrics
