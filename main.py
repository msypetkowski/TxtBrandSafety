#!/usr/bin/env python3

import json
import gensim
import nltk
from itertools import product
from nltk.corpus import wordnet
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import normalize


VECTORS_FILE = './GoogleNews-vectors-negative300.bin'
# VECTORS_FILE = '/tmp/GoogleNews-vectors-negative300.bin'


class WordModel:
    def __init__(self, model_type='wordnet'):
        self._model_type = model_type
        if model_type=='GoogleNews':
            self._model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
        elif model_type=='wordnet':
            pass
        else:
            raise ValueError("Unknown word model.")


    def similarity(self, word1, word2):
        """ Returns similarity of 2 words
        """
        if self._model_type == 'wordnet':
            syns1 = wordnet.synsets(word1)
            syns2 = wordnet.synsets(word2)
            ret =  [wordnet.wup_similarity(s1, s2) or 0
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
            return [w for w in words if len(wordnet.synsets(w)) > 0]
        else:
            ret = []
            for word in words:
                try:
                    self._model[word]
                    ret.append(word)
                except KeyError:
                    pass
            return ret


def tokens_to_feature_vector(tokens, keywords, word_model):
    return [word_model.calc_content_in_text(keyword, tokens) for keyword in keywords]


def read_metadata():
    """ Returns tuple of lists of strings (adtypes, keywords, sitetypes)
    """
    with open('data/keywords.json', 'r') as file:
        data = json.loads(file.read())
    return data['adtypes'], data['keywords'], data['sitetypes']


def prepare_dataset(word_model):
    """ Returns tuple (labels, feature_vectors, attributes_means, attributes_stds)
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

    labels, features = zip(*rows)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    features = (features - means) / stds
    assert len(labels) == len(features)
    assert len(means) == len(stds) == len(keywords)
    return labels, np.array(features), means, stds


def main():
    word_model = WordModel(model_type='wordnet')
    word_model = WordModel(model_type='GoogleNews')
    labels, features, means, stds = prepare_dataset(word_model)
    print('training examples count:', len(labels))
    # print('\n'.join(map(str,features)))

    print('attribute means values by class:')
    d1 = np.array([v for c,v in zip(labels, features) if c==0])
    d2 = np.array([v for c,v in zip(labels, features) if c==1])
    d3 = np.array([v for c,v in zip(labels, features) if c==2])
    print(np.mean(d1, 0))
    print(np.mean(d2, 0))
    print(np.mean(d3, 0))
    plt.scatter(d1[:, 0], d1[:, 6], s=80, c=d1[:, 12], marker='+')
    plt.scatter(d2[:, 0], d2[:, 6], s=80, c=d2[:, 12], marker='>')
    plt.scatter(d3[:, 0], d3[:, 6], s=80, c=d3[:, 12], marker=(5,0))
    plt.tight_layout()
    plt.show()
    print()

    # train classifier
    classifier = svm.SVC(decision_function_shape='ovo', probability=True)
    # classifier = GaussianNB()
    classifier.fit(features, labels)
    predicted_labels = np.argmax(classifier.predict_proba(features), axis=1)
    print('training accuracy:',
          np.mean(predicted_labels == np.array(labels)))
    print('training accuracy (not proba argmax):',
          np.mean(classifier.predict(features) == np.array(labels)))

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
        feature_vector = (feature_vector - means) / stds
        print("predicted probabilities:", classifier.predict_proba([feature_vector]))
        print("predicted class:", classifier.predict([feature_vector]))


if __name__ == '__main__':
    main()
