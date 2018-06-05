""" Main functionality of the whole system.
For given html
"""
import numpy as np
from core.html_to_texts import extract_texts_from_html


def classify_website(site, text_classifier):
    """ Returns vector of class probabilities
    """
    site = extract_texts_from_html(site)
    print('Website text blocks count:', len(site))
    site = ' '.join(site)
    print('Website text length:', len(site))
    return text_classifier.classify_texts([site])[0]


def measure_compatibility(probabilities, cost_matrix):
    """ Return vector of compabilities with given html
    for all ad types (see data/metadata.json).
    Compatibility can be nagative, but is always <= 1.
    """
    return 1 - (np.array(cost_matrix) @ probabilities)
