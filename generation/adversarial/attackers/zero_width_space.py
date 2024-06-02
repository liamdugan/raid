import random


class ZeroWidthSpaceAttack:
    def __init__(self, N=1):
        """
        This class takes a piece of text and inserts zero-width space characters within the text.

        Args:
            N (float): Between 0 and 1, indicating the fraction of characters where the space should be inserted
        """
        self.N = N
        self.zero_width_space = "\u200b"

    def attack(self, text):
        """
        This function inserts zero-width space in the text.

        Args:
            text (str): String containing the text to be modified

        Returns:
            str: Modified text with zero-width space inserted between characters
        """
        text = list(text)  # Convert to list because python strings are immutable

        # Get the number of zero-width spaces to be added based on the percentage N
        num_spaces = int(self.N * len(text))

        # Randomly sample the positions in the text to insert spaces
        indices_to_alter = random.sample(range(len(text)), num_spaces)

        # Randomly select the positions in the text to insert the spaces
        edits = []
        for i in sorted(indices_to_alter):
            edits.append((len("".join(text[: i + 1])), len("".join(text[: i + 1])) + 1))
            text[i] += self.zero_width_space

        return {"generation": "".join(text), "num_edits": len(edits), "edits": edits}
