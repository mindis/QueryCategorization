
import nltk
from nltk.stem import WordNetLemmatizer


def lemmatize(string):
    wl = WordNetLemmatizer()
    lem_w = [wl.lemmatize(w) for w in string.split(' ')]
    return ' '.join(lem_w)
