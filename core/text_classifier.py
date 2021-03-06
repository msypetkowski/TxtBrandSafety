import json
import pickle

import numpy as np
# from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

from . import read_metadata


class TextClassifier:
    def __init__(self, word_model, filename=None):
        self._word_model = word_model
        self._classifier = None
        if filename is None:
            metadata = read_metadata()
            self._metadata = (metadata['adtypes'], metadata['keywords'], metadata['sitetypes'])
            self._dataset = self.prepare_dataset()
            print('training examples count:', len(self._dataset[0]))
            # print('\n'.join(map(str,features)))
        else:
            self.load(filename)

    def tokens_to_feature_vector(self, tokens, keywords):
        return [self._word_model.calc_content_in_text(keyword, tokens) for keyword in keywords]

    def prepare_dataset(self):
        """ Returns tuple (labels, feature_vectors, attributes_means, attributes_stds)
        """
        rows = []
        _, keywords, sitetypes = self._metadata
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
                tokens = [self._word_model.tokenize(t) for t in texts]
                print('\t\t', 'average text tokens count', np.mean([len(t) for t in tokens]))

            # one dataset row for one text
            for tk in tokens:
                feature_vector = self.tokens_to_feature_vector(tk, keywords)
                rows.append((class_id, feature_vector))

        labels, features = zip(*rows)
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        features = (features - means) / stds
        assert len(labels) == len(features)
        assert len(means) == len(stds) == len(keywords)
        return labels, np.array(features), means, stds

    def train_classifier(self, classifier_type='svm'):
        if classifier_type == 'svm':
            self._classifier = svm.SVC(decision_function_shape='ovo', probability=True)
        elif classifier_type == 'nb':
            self._classifier = GaussianNB()
        else:
            raise ValueError("Unsupported classifier type.")

        labels, features, means, stds = self._dataset
        print('training examples count:', len(labels))

        print('attribute means values by class:')
        d1 = np.array([v for c, v in zip(labels, features) if c == 0])
        d2 = np.array([v for c, v in zip(labels, features) if c == 1])
        d3 = np.array([v for c, v in zip(labels, features) if c == 2])
        print(np.mean(d1, 0))
        print(np.mean(d2, 0))
        print(np.mean(d3, 0))
        # plt.scatter(d1[:, 0], d1[:, 6], s=80, c=d1[:, 12], marker='+')
        # plt.scatter(d2[:, 0], d2[:, 6], s=80, c=d2[:, 12], marker='>')
        # plt.scatter(d3[:, 0], d3[:, 6], s=80, c=d3[:, 12], marker=(5, 0))
        # plt.tight_layout()
        # plt.show()
        print()

        # train classifier
        self._classifier.fit(features, labels)
        predicted_labels = np.argmax(self._classifier.predict_proba(features), axis=1)
        print('training accuracy (proba argmax):',
              np.mean(predicted_labels == np.array(labels)))
        print('training accuracy (normal svm):',
              np.mean(self._classifier.predict(features) == np.array(labels)))

    def cross_validate(self, classifier_type='svm'):
        if classifier_type == 'svm':
            self._classifier = svm.SVC(decision_function_shape='ovo', probability=True)
        elif classifier_type == 'nb':
            self._classifier = GaussianNB()
        else:
            raise ValueError("Unsupported classifier type.")

        labels, features, means, stds = self._dataset
        # features = features * stds + means

        accuracy = cross_val_score(self._classifier, features, labels, cv=5)
        print(f'Accuracy: {np.mean(accuracy)}')
        return accuracy

    def classify_texts(self, texts):
        """ Returns vector of probabilities.
        """
        _, keywords, _ = self._metadata
        _, _, means, stds = self._dataset
        tokens = [self._word_model.tokenize(text) for text in texts]
        feature_vectors = np.array([self.tokens_to_feature_vector(tk, keywords)
                                    for tk in tokens])
        feature_vectors = (feature_vectors - means) / stds
        return self._classifier.predict_proba(feature_vectors)

    def save(self, filename='classifier_data'):
        with open(filename, 'wb') as file:
            pickle.dump((self._metadata, self._dataset, self._classifier), file)

    def load(self, filename='classifier_data'):
        with open(filename, 'rb') as file:
            self._metadata, self._dataset, self._classifier = pickle.load(file)
