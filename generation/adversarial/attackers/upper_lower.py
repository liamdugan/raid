import random

import nltk


class UpperLowerFlipAttack:
    def __init__(self, N=0.5):
        """
        This class takes a piece of text and flips the case of the first letter of words based on N%.
        Most effective on articles.

        Args:
            N (float): Between 0 and 1, indicating the percentage of first letters of words to be altered
        """
        self.N = N
        self.tokenizer = nltk.tokenize.NLTKWordTokenizer()

    def attack(self, text):
        # Get all indices of starts of token spans where the first char is alphabetic
        indices = [s for s, e in self.tokenizer.span_tokenize(text) if text[s].isalpha()]

        # Get the number of indices to be changed based on the percentage N
        num_to_flip = int(len(indices) * self.N)

        # Select num_to_flip indices randomly from the candidate indices
        flip_indices = random.sample(indices, num_to_flip)

        # Flip indices to upper if lower and to lower if upper
        text = list(text)  # Cast to list since python strings are immutable
        for i in flip_indices:
            text[i] = text[i].lower() if text[i].isupper() else text[i].upper()

        # Get the spans for the flipped indices to be consistent with the format
        edits = [(i, i + 1) for i in flip_indices]

        return {"generation": "".join(text), "num_edits": len(edits), "edits": edits}
