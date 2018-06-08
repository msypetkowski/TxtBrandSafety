import json


def read_metadata():
    """ Returns tuple of lists of strings (adtypes, keywords, sitetypes)
    """
    with open('data/metadata.json', 'r') as file:
        data = json.loads(file.read())
    return data


from .measure_compatibility import measure_compatibility, classify_website
from .text_classifier import TextClassifier
from .word_model import WordModel


__all__ = [WordModel, TextClassifier, classify_website, measure_compatibility, read_metadata]
