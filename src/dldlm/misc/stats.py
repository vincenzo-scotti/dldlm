from math import log10

from typing import List, Set
from collections import Counter

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp.add_pipe('spacytextblob')

DATA_SPECIFIC_STOPWORDS = {"like", "know", "yeah", "think", "feel", "going", "mean", "kind", "right", "things", "want",
                           "way", "time", "people", "sort", "thing", "good", "said", "okay", "guess", "maybe", "lot",
                           "oh", "little", "got", "work", "feeling", "actually", "need", }
PUNCTUATION: Set[str] = set("[!\"#$%&()*+,-./:;<=>?@[]\\^`{|}~_']") | {'...', '``', '\'\'', '--'}
STOP_WORDS: Set[str] = set(stopwords.words('english')) | spacy_nlp.Defaults.stop_words | PUNCTUATION


def get_word_counts(s: str, remove_stopwords: bool = False) -> Counter:
    return Counter(
        w for w in (w.lower() for w in word_tokenize(s))
        if w not in (STOP_WORDS if remove_stopwords else PUNCTUATION)
    )


def get_word_document_counts(docs: List[str], remove_stopwords: bool = False):
    return Counter(
        w for s in docs for w in set(w.lower() for w in word_tokenize(s))
        if w not in (STOP_WORDS if remove_stopwords else PUNCTUATION)
    )


def tf_idf(word_counts: Counter, word_document_counts: Counter, n_docs: int, min: int = 0) -> Counter:
    # Get number of words in sample
    n_words = sum(word_counts.values())
    # Compute TF IDF
    return Counter({
        word: (word_counts[word] / n_words) * log10(n_docs / word_document_counts[word])
        for word in word_counts if word_document_counts[word] > min
    })


def sentiment(sample: str) -> str:
    # Helper function to convert polarity to categorical sentiment
    def encode_sentiment(polarity: float) -> str:
        if polarity < -0.6:
            return 'Very negative'
        elif polarity < -0.2:
            return 'Negative'
        elif polarity < 0.2:
            return 'Neutral'
        elif polarity < 0.6:
            return 'Positive'
        else:
            return 'Very Positive'

    return encode_sentiment(spacy_nlp(sample)._.blob.polarity)
