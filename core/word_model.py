from itertools import product

import gensim
import nltk
import numpy as np
from nltk.corpus import wordnet

VECTORS_FILE = './GoogleNews-vectors-negative300.bin'


class WordModel:
    def __init__(self, model_type='wordnet'):
        self._model_type = model_type
        if model_type == 'GoogleNews':
            self._model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
        elif model_type == 'wordnet':
            pass
        else:
            raise ValueError("Unknown word model.")

    def similarity(self, word1, word2):
        """ Returns similarity of 2 words
        """
        if self._model_type == 'wordnet':
            syns1 = wordnet.synsets(word1)
            syns2 = wordnet.synsets(word2)
            ret = [wordnet.wup_similarity(s1, s2) or 0
                   for s1, s2 in product(syns1, syns2)]
            return np.mean(ret)
        else:
            return self._model.similarity(word1, word2)

    def calc_content_in_text(self, word, text):
        # TODO: consider different method
        res = 0
        for w in text:
            res += self.similarity(w, word)
        return res / len(text)

    def tokenize(self, text):
        words = nltk.word_tokenize(text)
        if self._model_type == 'wordnet':
            ret = [w for w in words if len(wordnet.synsets(w)) > 0]
        else:
            ret = []
            for word in words:
                try:
                    self._model[word]
                    ret.append(word)
                except KeyError:
                    pass
        if len(ret) < 1:
            ret = ['nothing']
        return ret
