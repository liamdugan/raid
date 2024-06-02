import random

import nltk


class InsertParagraphsAttack:
    def __init__(self, N=0.5):
        """
        This class takes a piece of text and adds double newlines randomly between sentences.

        Args:
            N (float): Between 0 and 1, indicating the percentage of sentences where double newlines will be added
        """
        self.N = N

    def attack(self, text):
        # Tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)

        # Get the number of sentences to alter as N * total amount of inter-sentence spaces
        sentences_to_alter = int((len(sentences) - 1) * self.N)

        # Randomly sample which indices to append double newlines to
        indices_to_alter = random.sample(range(1, len(sentences)), sentences_to_alter)

        # Prepend double newline to each of the sampled indices
        edits = []
        for i in sorted(indices_to_alter):
            sentences[i] = "\n\n" + sentences[i]
            edit_start = len(" ".join(sentences[:i])) + 1
            edits.append((edit_start, edit_start + 2))

        return {"generation": " ".join(sentences), "num_edits": len(edits), "edits": edits}
