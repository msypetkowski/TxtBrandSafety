import gensim
import numpy as np


VECTORS_FILE = './GoogleNews-vectors-negative300.bin'
# VECTORS_FILE = '/tmp/GoogleNews-vectors-negative300.bin'


class Model:
    def __init__(self):
        self._model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)

    def get_embeddings(self, words):
        """ Returns list of embeddings.
        :param words: list of strings
        """
        return [np.array(self._model[w]) for w in words]

    def similarity(self, embedding1, embedding2):
        """ Returns similarity of 2 embeddings
        (lower value means larger similarity)
        """
        return np.linalg.norm(e1-e2)


if __name__ == '__main__':
    model = Model()
    words = ['beer', 'child', 'children', 'toys', 'weapons', 'gun']
    embeddings = model.get_embeddings(words)
    for e1 in embeddings:
        for e2 in embeddings:
            norm = round(m.similarity(e1, e2), 2)
            print((str(norm) + ',').ljust(8), end='')
        print()
