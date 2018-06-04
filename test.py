#!/usr/bin/env python3
"""
Main functionality testing script.
Check ad compatibility for a few webites.
"""

import numpy as np
import urllib.request

from html_to_texts import classify_website
from train_text_classifier import TextClassifier, read_metadata
from word_model import WordModel
from measure_compatibility import measure_compatibility


def fetch_url(url):
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}
    req = urllib.request.Request(url, headers=hdr)
    site = urllib.request.urlopen(req).read()
    return site


def simple_test():
    metadata = read_metadata()
    cost_matrix = np.array(metadata['costmatrix'])
    print("website classes:", metadata['sitetypes'])
    print("ad types:", metadata['adtypes'])

    links = ['http://studia.elka.pw.edu.pl/EN/',
             'http://www.premierwines.com/',
             'https://www.tripadvisor.com/Restaurant_Review-g29220-d5768632-Reviews-Hali_imaile_General_Store-Maui_Hawaii.html',
             'https://www.overstock.com/Electronics/2/store.html']

    word_model = WordModel(model_type='GoogleNews')
    text_classifier = TextClassifier(word_model, filename='classifier_data')

    for l in links:
        print('-'*50)
        html = fetch_url(l)
        print(l)
        probabilities = np.array(classify_website(html, text_classifier))
        print('class probabilities:', probabilities)
        print('ads compatibilities:', measure_compatibility(probabilities, cost_matrix))


if __name__ == '__main__':
    simple_test()
