import json
# import gensim
import nltk
from itertools import product
from nltk.corpus import wordnet
import numpy as np

from sklearn import svm
from sklearn.naive_bayes import GaussianNB


# VECTORS_FILE = './GoogleNews-vectors-negative300.bin'
# VECTORS_FILE = '/tmp/GoogleNews-vectors-negative300.bin'


class WordModel:
    def __init__(self):
        pass
        # self._model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)

    def similarity(self, word1, word2):
        """ Returns similarity of 2 words
        """
        syns1 = wordnet.synsets(word1)
        syns2 = wordnet.synsets(word2)
        ret =  max((wordnet.wup_similarity(s1, s2) or 0, s1, s2)
                for s1, s2 in product(syns1, syns2))
        return ret[0]
        # return self._model.similarity(word1, word2)

    def calc_content_in_text(self, word, text):
        # TODO: consider different method
        res = 0
        for w in text:
            res += self.similarity(w, word)
        return res / len(text)

    def tokenize(self, text):
        ret = nltk.word_tokenize(text)
        return [w for w in ret if len(wordnet.synsets(w)) > 0]
        # words = text.split()
        # ret = []
        # for word in words:
        #     try:
        #         self._model[word]
        #         ret.append(word)
        #     except KeyError:
        #         pass
        # return ret


def tokens_to_feature_vector(tokens, keywords, word_model):
    return [word_model.calc_content_in_text(keyword, tokens) for keyword in keywords]


def read_metadata():
    """ Returns tuple of lists of strings (adtypes, keywords, sitetypes)
    """
    with open('data/keywords.json', 'r') as file:
        data = json.loads(file.read())
    return data['adtypes'], data['keywords'], data['sitetypes']


def prepare_dataset(word_model):
    """ Returns list of pairs (int:class, list:feature_vector)
    """
    rows = []
    _, keywords, sitetypes = read_metadata()
    print('keywords:')
    print('\t', keywords)
    print('dataset:')

    # read texts for each site type
    for class_id, site_type in enumerate(sitetypes):
        print('\t', 'class_id:', class_id)
        print('\t', 'site_type:', site_type)
        with open('data/' + site_type + '.json', 'r') as file:
            texts = json.loads(file.read())
            print('\t\t', 'texts count in dataset:', len(texts))
            tokens = [word_model.tokenize(t) for t in texts]
            print('\t\t', 'average text tokens count', np.mean([len(t) for t in tokens]))

        # one dataset row for one text
        for tk in tokens:
            feature_vector = tokens_to_feature_vector(tk, keywords, word_model)
            rows.append((class_id, feature_vector))

    return rows


def main():
    word_model = WordModel()
    dataset = prepare_dataset(word_model)
    print('training examples count:', len(dataset))
    # print('\n'.join(map(str,dataset)))

    print('attribute means values by class:')
    print(np.mean([v for c,v in dataset if c==0], 0))
    print(np.mean([v for c,v in dataset if c==1], 0))
    print()

    # train classifier
    # classifier = svm.SVC()
    classifier = GaussianNB()
    labels, features = zip(*dataset)
    classifier.fit(features, labels)
    print('training accuracy:', np.mean(np.array(classifier.predict(features)) == np.array(labels)))

    # For testing similarity function
    #
    # print()
    # words = ['beer', 'child', 'children', 'toys', 'weapons', 'gun']
    # for w1 in words:
    #     for w2 in words:
    #         norm = round(word_model.similarity(w1, w2), 2)
    #         print((str(norm) + ',').ljust(8), end='')
    #     print()

    _, keywords, _ = read_metadata()
    while True:
        tokens = word_model.tokenize(input('Enter text to classify:'))
        feature_vector = tokens_to_feature_vector(tokens, keywords, word_model)
        print("predicted class:", classifier.predict([feature_vector]))


if __name__ == '__main__':
    main()
