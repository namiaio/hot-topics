import re
import hfst

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('finnish')

lemmatizer = hfst.HfstInputStream("omorfi.describe.hfst").read().lookup
tokenizer = RegexpTokenizer(r'[a-zA-ZåäöÅÄÖ\']+').tokenize
stem = SnowballStemmer('finnish').stem


def clusterTokenizer(text):
    return tokenize(text, stem=True)


def topicTokenizer(text):
    return tokenize(text, stem=False)


def tokenize(text, **kwargs):
    result = []
    words = tokenizer(text)

    for word in words:
        lemmed = lemmatizer(word)

        if len(lemmed) > 0:
            analyzed = hfst_result_parser(lemmed)
        else:
            analyzed = [word]

        if kwargs['stem']:
            result = result + [stem(s) for s in analyzed]
        else:
            result = result + [''.join(analyzed)]

    return result


def hfst_result_parser(result):
    words = []
    last = len(result)-1
    s = result[last][0]

    lemmas = re.compile(r"\[WORD_ID=([^]]*)\]").finditer(s)

    for lemma in lemmas:
        word = lemma.group(1).split("_")[0]
        words.append(word)

    return words
