#!/usr/bin/env python3

import os
import sys

import numpy as np

from core.text_classifier import TextClassifier
from core.word_model import WordModel


def main():
    filename = 'classifier_data' if os.path.exists('classifier_data') else None
    word_model = WordModel(model_type='GoogleNews')
    text_classifier = TextClassifier(word_model, filename=filename)
    if len(sys.argv) == 2 and sys.argv[1] == 'validate':
        text_classifier.cross_validate(classifier_type='svm')

    else:
        text_classifier.train_classifier(classifier_type='svm')
        text_classifier.save()
        # while True:
        #     text = input('Enter text to classify:')

        #     probabilities = text_classifier.classify_texts([text])[0]
        #     print("predicted probabilities:", probabilities)
        #     print("probabilities argmax:", np.argmax(probabilities))


if __name__ == '__main__':
    main()
