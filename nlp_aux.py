import math
import wikipedia
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk.data import find

nltk.download('wordnet')
nltk.download('omw')
nltk.download('stopwords')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('treebank')

import multiprocessing
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews, treebank


def wiki_bag_of_words(page, n=5, remove_stop_words=False, print_bow=False):
    wikipedia.set_lang("it")
    page = wikipedia.page(page)
    counter = Counter()
    words = page.content.lower().split()
    counter.update(words)
    if remove_stop_words:
        for stopword in stopwords.words("italian"):
            del counter[stopword]

    if print_bow:
        for letter, count in counter.most_common(n):
            print('%s: %7d' % (letter, count))
    return counter


def bow_distance(vec1, vec2):
    a = [vec1, vec2]
    d = set()
    for row in a:
        d = d.union(row.keys())
    dist = 0
    for k in d:
        dist += (vec1[k] - vec2[k]) ** 2
    return math.sqrt(dist)


def _lang_ratios(text):
    lang_ratios = {}  # initialize a dictionary called lang_ratios
    tokens = wordpunct_tokenize(text)  # tokenize text
    words = [word.lower() for word in tokens]  # convert to lower case
    words_set = set(words)

    for language in stopwords.fileids():  # select a language from the list of available languages
        stopwords_set = set(stopwords.words(language))
        common_set = words_set.intersection(stopwords_set)
        lang_ratios[language] = len(common_set)

    return lang_ratios


def detect_language(text):
    ratios = _lang_ratios(text)
    lang = max(ratios,
               key=ratios.get)  # the key option takes a function returns the key having the largest "value" in the iterable
    return lang

from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re

def processSentence2(sentence, lemmatizer):
    ww = sentence.split()
    #ww = nltk.word_tokenize(sentence)
    ww = [x for x in ww if x not in stopwords.words("english")]
    ww = [x for x in ww if x !=""]
    ww = [lemmatizer.lemmatize(x) for x in ww]
    if len(ww)>2:
        return " ".join(ww)
    else:
        return np.nan

def prepare_w2v():
    mr = Word2Vec(movie_reviews.sents())
    t = Word2Vec(treebank.sents())
    return mr, t


def prepare_simpsons():
    lemmatizer = WordNetLemmatizer()

    df = pd.read_csv('simpsons_dataset.csv')
    df.columns = ["name", "spoken_words"]
    df.spoken_words = [re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words']]
    df.spoken_words = [re.sub("[/']", '', str(row)).lower() for row in df.spoken_words]
    cleaned2 = [processSentence2(x, lemmatizer) for x in df.spoken_words.tolist()]
    words22 = [x for x in cleaned2 if str(x) != "nan"]
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    words33 = [x.split() for x in words22]
    w2v_model = Word2Vec(min_count=20,
                         window=5,
                         vector_size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1,
                         sg=1)

    w2v_model.build_vocab(words33, progress_per=10000)
    #vocab_size = len(w2v_model.wv.vocab)
    w2v_model.train(words33, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)
    w2v_model.init_sims(replace=True)
    return w2v_model