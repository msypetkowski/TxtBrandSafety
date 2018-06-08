import numpy as np

from . import MAX_WEBSITE_SIZE, MANAGING_PORT
from .connection import Connection
from core import TextClassifier
from core import WordModel
from core import classify_website
from core import measure_compatibility
from core import read_metadata

class WorkerAgent:

    def __init__ (self, front_addr):
        metadata = read_metadata()
        self._cost_matrix = np.array(metadata['costmatrix'])
        self._site_types = metadata['sitetypes']
        self._ad_types = metadata['adtypes']
        word_model = WordModel(model_type='GoogleNews')
        self._text_classifier = TextClassifier(word_model, filename='classifier_data')
        self._conn = Connection()
        self._conn.connect(front_addr, MANAGING_PORT)
        self._metadata = metadata

    def main_loop(self):
        if not self._conn.is_valid():
            print("Broken connection")
            return
        print('starting main loop')
        while True:
            html = self._conn.receive()
            print("----------new task")
            if not self._conn.is_valid():
                print("Broken connection")
                return
            probabilities = np.array(classify_website(html, self._text_classifier))
            proposed_ads = measure_compatibility(probabilities, self._cost_matrix)
            # proposed_ads = [1,2,3,4,5]
            print("probabilities:", probabilities)
            data = {adtype:r for adtype, r in zip(self._metadata["adtypes"], proposed_ads)}
            print("proposed_ads:", data)
            self._conn.send(str(data).encode())
            if not self._conn.is_valid():
                print("Broken connection")
                return
