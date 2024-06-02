import random


class HomoglyphAttack:
    def __init__(self, N=1.0):
        self.mapping = {
            "a": ["а"],
            "A": ["А", "Α"],
            "B": ["В", "Β"],
            "e": ["е"],
            "E": ["Е", "Ε"],
            "c": ["с"],
            "p": ["р"],
            "K": ["К", "Κ"],
            "O": ["О", "Ο"],
            "P": ["Р", "Ρ"],
            "M": ["М", "Μ"],
            "H": ["Н", "Η"],
            "T": ["Т", "Τ"],
            "X": ["Х", "Χ"],
            "C": ["С"],
            "y": ["у"],
            "o": ["о"],
            "x": ["х"],
            "I": ["І", "Ι"],
            "i": ["і"],
            "N": ["Ν"],
            "Z": ["Ζ"],
        }

    def attack(self, text):
        text = list(text)
        edits = []
        for i, char in enumerate(text):
            if char in self.mapping:
                text[i] = random.choice(self.mapping[char])
                edits.append((i, i + 1))

        return {"generation": "".join(text), "num_edits": len(edits), "edits": edits}
