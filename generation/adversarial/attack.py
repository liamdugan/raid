from attackers.alter_number import AlterNumbersAttack
from attackers.alternative_spelling import AlternativeSpellingAttack
from attackers.article_deletion import ArticleDeletionAttack
from attackers.homoglyph import HomoglyphAttack
from attackers.insert_paragraphs import InsertParagraphsAttack
from attackers.paraphrase import ParaphraseAttack
from attackers.perplexity_misspelling import PerplexityMisspellingAttack
from attackers.synonym import SynonymAttack
from attackers.upper_lower import UpperLowerFlipAttack
from attackers.whitespace import WhiteSpaceAttack
from attackers.zero_width_space import ZeroWidthSpaceAttack


class Attack:
    """Shared interface for all adversarial attacks"""

    def attack(self, text: str, **kwargs) -> dict:
        """
        Takes in a text and outputs a dictionary with the key 'generation' pointing
        to the adversarially modified text and any number of other keys containing metrics
        """
        pass


def get_attack(attack_name: str) -> Attack:
    if attack_name == "homoglyph":
        return HomoglyphAttack(1.0)
    elif attack_name == "number":
        return AlterNumbersAttack(0.5)
    elif attack_name == "article_deletion":
        return ArticleDeletionAttack(0.5)
    elif attack_name == "insert_paragraphs":
        return InsertParagraphsAttack(0.5)
    elif attack_name == "perplexity_misspelling":
        return PerplexityMisspellingAttack(0.2)
    elif attack_name == "upper_lower":
        return UpperLowerFlipAttack(0.05)
    elif attack_name == "whitespace":
        return WhiteSpaceAttack(0.2)
    elif attack_name == "zero_width_space":
        return ZeroWidthSpaceAttack(1.0)
    elif attack_name == "synonym":
        return SynonymAttack(0.5)
    elif attack_name == "paraphrase":
        return ParaphraseAttack(1.0)
    elif attack_name == "alternative_spelling":
        return AlternativeSpellingAttack(1.0)
    else:
        raise ValueError("Invalid attacker name")
