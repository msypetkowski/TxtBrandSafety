#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize

from word_model import WordModel


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
    # word_model = WordModel(model_type='wordnet')
    word_model = WordModel(model_type='GoogleNews')
    labels, features, means, stds = prepare_dataset(word_model)
    print('training examples count:', len(labels))
    # print('\n'.join(map(str,features)))

    print('attribute means values by class:')
    d1 = np.array([v for c, v in zip(labels, features) if c == 0])
    d2 = np.array([v for c, v in zip(labels, features) if c == 1])
    d3 = np.array([v for c, v in zip(labels, features) if c == 2])
    print(np.mean(d1, 0))
    print(np.mean(d2, 0))
    print(np.mean(d3, 0))
    plt.scatter(d1[:, 0], d1[:, 6], s=80, c=d1[:, 12], marker='+')
    plt.scatter(d2[:, 0], d2[:, 6], s=80, c=d2[:, 12], marker='>')
    plt.scatter(d3[:, 0], d3[:, 6], s=80, c=d3[:, 12], marker=(5, 0))
    plt.tight_layout()
    plt.show()
    print()

    # train classifier
    classifier = svm.SVC(decision_function_shape='ovo', probability=True)
    # classifier = GaussianNB()
    classifier.fit(features, labels)
    predicted_labels = np.argmax(classifier.predict_proba(features), axis=1)
    print('training accuracy (proba argmax):',
          np.mean(predicted_labels == np.array(labels)))
    print('training accuracy (normal svm):',
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
        if len(tokens) < 1:
            print("Invalid text")
            continue
        feature_vector = tokens_to_feature_vector(tokens, keywords, word_model)
        feature_vector = (feature_vector - means) / stds
        probabilities = classifier.predict_proba([feature_vector])[0]
        print("predicted probabilities:", probabilities)
        print("probabilities argmax:", np.argmax(probabilities))
        print("predicted class:", classifier.predict([feature_vector])[0])


if __name__ == '__main__':
    main()
