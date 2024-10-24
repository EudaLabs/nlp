from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)