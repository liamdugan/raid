import random

import regex as re
import requests


class AlternativeSpellingAttack:
    def __init__(self, N=1.0):
        """
        This class takes a piece of text and swaps words with their british spellings.
        """
        self.N = N

        url = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
        self.us_to_gb_spelling = requests.get(url).json()

    def attack(self, text):
        # Find all instances of american spelling in the text (if overlaps, default to longest match)
        matches = list(re.finditer(r"\L<words>", text, words=self.us_to_gb_spelling.keys()))

        # Get number of matches to swap as proportion of N
        number_of_edits = int(len(matches) * self.N)

        # Randomly sample the matches to swap
        matches_to_swap = random.sample(matches, number_of_edits)

        delta = 0
        edits = []
        for m in sorted(matches_to_swap, key=lambda m: m.start()):
            # Get the left and right index of the swap
            left = m.start() + delta
            right = m.end() + delta

            # Get the match and insert it into the text
            gb_spelling = self.us_to_gb_spelling[m.group()]
            text = text[:left] + gb_spelling + text[right:]

            # Edit delta to account for indexing changes
            delta += len(gb_spelling) - len(m.group())

            # Add the edited span to edits
            edits.append((left, left + len(gb_spelling)))

        return {"generation": text, "num_edits": len(edits), "edits": edits}
