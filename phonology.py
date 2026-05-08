import pronouncing
import re


def get_phonology_for_word(word):
    phones = pronouncing.phones_for_word(word)

    if not phones:
        return {
            "n_phonemes": None,
            "n_syllables": None,
            "onset_phoneme": None
        }

    p = phones[0]
    toks = p.split()

    return {
        "n_phonemes": len(toks),
        "n_syllables": pronouncing.syllable_count(p),
        "onset_phoneme": re.sub(r"\d", "", toks[0])
    }