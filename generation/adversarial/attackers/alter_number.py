import random
import re


class AlterNumbersAttack:
    def __init__(self, N=0.5):
        """
        This class takes a piece of text and alters numbers within the sentences.

        Args:
            N (float): Between 0 and 1, indicating the percentage of numbers to be altered
        """
        self.N = N

    def attack(self, text):
        """
        This function replaces numbers in the text with random integers of the same order.

        Args:
            text (str): String containing the text to be modified

        Returns:
            str: Modified text with some numbers replaced with random numbers
        """
        # Find all numbers in the text
        matches = [m.span() for m in re.finditer("\d+\.?\d*", text)]

        # Get the total number of numbers to alter
        number_of_edits = int(len(matches) * self.N)

        # Randomly select which numbers to alter
        spans_to_alter = sorted(random.sample(matches, number_of_edits))

        # For each of the spans to alter, randomly sample digits to replace existing ones
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        text = list(text)
        for start, end in spans_to_alter:
            for i in range(start, end):
                text[i] = random.choice(digits) if text[i].isnumeric() else text[i]

        return {"generation": "".join(text), "num_edits": len(spans_to_alter), "edits": spans_to_alter}
