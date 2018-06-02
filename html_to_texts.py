#!/usr/bin/env python3

import re
import urllib.request
    # import urllib2,cookielib

from train_text_classifier import TextClassifier
from word_model import WordModel


def extract_texts_from_html(site):
    txt = re.sub("(?s)<[^>]*>(\\s*<[^>]*>)*", r" ", site.decode()).replace('-->', '\n\n')
    return txt.strip()


def classify_website(site, text_classifier):
    site = extract_texts_from_html(site)
    print('Website text length:', len(site))
    # print(site)
    return text_classifier.classify_texts([site])[0]


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


def main():
    word_model = WordModel(model_type='GoogleNews')
    text_classifier = TextClassifier(word_model, filename='classifier_data')

    html = fetch_url('http://studia.elka.pw.edu.pl/EN/')
    print(classify_website(html, text_classifier))

    html = fetch_url('http://www.premierwines.com/')
    print(classify_website(html, text_classifier))

    html = fetch_url('https://www.tripadvisor.com/Restaurant_Review-g29220-d5768632-Reviews-Hali_imaile_General_Store-Maui_Hawaii.html')
    print(classify_website(html, text_classifier))

if __name__ == '__main__':
    main()
