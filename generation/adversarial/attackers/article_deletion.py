import random


class ArticleDeletionAttack:
    def __init__(self, N=0.5):
        """
        This class takes a piece of text and randomly deletes articles from the sentences.

        Args:
            N (float): Between 0 and 1, indicating the percentage of articles to be deleted
        """
        self.N = N
        self.articles = [" a ", " an ", " the "]

    def find_article_indices(self, text, article):
        start = 0
        while start < len(text):
            start = text.find(article, start)
            if start == -1:
                break
            yield start
            start += len(article) - 1

    def attack(self, text):
        """
        This function deletes articles in the text.

        Args:
            text (str): String containing the text to be modified

        Returns:
            str: Modified text with some articles deleted
        """

        # List of all indices where articles are found
        all_indices = [
            (index, article) for article in self.articles for index in self.find_article_indices(text, article)
        ]

        # Randomly select a subset of these indices based on N
        indices_to_delete = random.sample(all_indices, int(len(all_indices) * self.N))

        # Convert text to a list for easier character manipulation
        text = list(text)
        for index, article in sorted(indices_to_delete, reverse=True):
            # Delete the article in place (leaving the final space)
            # Loop from back to front to prevent indexing errors
            del text[index : index + len(article) - 1]

        # Get the indices in the original text where swaps were made
        total_deletions = 0
        edits = []
        for index, article in sorted(indices_to_delete, reverse=False):
            edits.append((index - total_deletions, index - total_deletions + 1))
            total_deletions += len(article) - 1

        return {"generation": "".join(text), "num_edits": len(edits), "edits": edits}
