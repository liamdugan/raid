import random


class WhiteSpaceAttack:
    def __init__(self, N=0.5):
        """
        This class takes a piece of text and randomly inserts whitespaces within the sentences.

        Args:
            N (float): Between 0 and 1, indicating the percentage of whitespaces to be altered
        """
        self.N = N

    def attack(self, text):
        # Split on spaces
        texts = text.split(" ")

        # Determine the number of spaces to insert
        spaces_to_alter = int(len(texts) * self.N)

        # Randomly sample indices to insert spaces at (with replacement!)
        indices_to_alter = random.choices(range(len(texts)), k=spaces_to_alter)

        # For all indices to alter, insert an extra space character after the token
        edits = []
        for i in sorted(indices_to_alter):
            edits.append((len(" ".join(texts[: i + 1])), len(" ".join(texts[: i + 1])) + 1))
            texts[i] += " "

        return {"generation": " ".join(texts), "num_edits": len(edits), "edits": edits}
