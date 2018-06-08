#!/usr/bin/env python3
"""
Main functionality testing script.
Check ad compatibility for a few webites.
"""

import urllib.request

import numpy as np
import json

from core import TextClassifier
from core import WordModel
from core import classify_website
from core import measure_compatibility
from core import read_metadata


def fetch_url(url):
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
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
        print('-' * 50)
        html = fetch_url(l)
        print(l)
        probabilities = np.array(classify_website(html, text_classifier))
        print('class probabilities:', probabilities)
        print('ads compatibilities:', measure_compatibility(probabilities, cost_matrix))


def benchmark(threshold):
    metadata = read_metadata()
    cost_matrix = np.array(metadata['costmatrix'])
    site_types = metadata['sitetypes']
    ad_types = metadata['adtypes']

    print("website classes:", site_types)
    print("ad types:", ad_types)
    print()

    with open('data/test_data.json', 'r') as file:
        test_data = json.loads(file.read())

    word_model = WordModel(model_type='GoogleNews')
    text_classifier = TextClassifier(word_model, filename='classifier_data')
    total_best_acc, total_banned_acc = [], []



    for site_type, site_dict in test_data.items():
        banned_ads = [ad_types.index(banned) for banned in site_dict['banned']]
        best_ad = ad_types.index(site_dict['best'])
        best_acc_tmp, banned_acc_tmp = 0, 0

        links = site_dict["links"]
        for l in links:
            html = fetch_url(l)
            probabilities = np.array(classify_website(html, text_classifier))
            proposed_ads = measure_compatibility(probabilities, cost_matrix)
            banned_pred = np.flatnonzero(proposed_ads < threshold) # indicies of banned ads
            best_ad_pred = np.argmax(proposed_ads)  # index of ad with highest score
            if proposed_ads[best_ad_pred] < threshold:
                best_ad_pred = -1
            best_acc_tmp += (best_ad_pred == best_ad)
            jaccard = len(np.intersect1d(banned_ads, banned_pred)) / len(np.union1d(banned_ads, banned_pred))
            banned_acc_tmp += jaccard

        site_banned_acc = banned_acc_tmp / len(links)
        site_best_acc = best_acc_tmp / len(links)
        total_banned_acc.append(site_banned_acc)
        total_best_acc.append(site_best_acc)

        print(f'Type: {site_type}')
        print(f'Best ad accuracy: {site_best_acc}')
        print(f'Banned ads accuracy: {site_banned_acc}')
        print()

    print(f'TOTAL:')
    print(f'Best ad accuracy: {np.mean(total_best_acc)}')
    print(f'Banned ads accuracy: {np.mean(total_banned_acc)}')
    print()             


if __name__ == '__main__':
    # simple_test()
    benchmark(threshold=0.25)
