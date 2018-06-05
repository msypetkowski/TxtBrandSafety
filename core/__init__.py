import json


def read_metadata():
    """ Returns tuple of lists of strings (adtypes, keywords, sitetypes)
    """
    with open('data/metadata.json', 'r') as file:
        data = json.loads(file.read())
    return data


from .word_model import WordModel
from .text_classifier import TextClassifier
from .measure_compatibility import measure_compatibility, classify_website

__all__ = [WordModel, TextClassifier, classify_website, measure_compatibility, read_metadata]
