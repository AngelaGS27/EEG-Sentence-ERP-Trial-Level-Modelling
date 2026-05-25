import wordfreq


def get_zipf_frequency(word):
    return wordfreq.zipf_frequency(word, "en")


def get_word_frequency(word):
    return wordfreq.word_frequency(word, "en")