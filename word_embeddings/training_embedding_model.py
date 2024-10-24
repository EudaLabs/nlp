#This file didn't complete yet.

from gensim.test.utils import datapath
from gensim import utils
import gensim.models

class MyCorpus:

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)


sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)

vec_same = model.wv['same']
print(vec_same[:10])

