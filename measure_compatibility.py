""" Main functionality of the whole system.
For given html 
"""
import numpy as np


def measure_compatibility(probabilities, cost_matrix):
    """ Return vector of compabilities with given html
    for all ad types (see data/metadata.json).
    Compatibility can be nagative, but is always <= 1.
    """

    return 1 - (np.array(cost_matrix) @ probabilities)
