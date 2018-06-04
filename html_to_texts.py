import re
import urllib.request
from bs4 import BeautifulSoup

from train_text_classifier import TextClassifier
from word_model import WordModel


def extract_texts_from_html(site):
    def additional_filter(site):
        txt = re.sub("(?s)<[^>]*>(\\s*<[^>]*>)*", r" ", site.replace('-->', '\n'))
        return txt.strip()
    soup = BeautifulSoup(site, "html.parser")
    data = soup.findAll(text=True)
    def visible(element):
        if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif re.match('<!--.*-->', str(element.encode('utf-8'))):
            return False
        return True
    result = filter(visible, data)
    result = [additional_filter(r) for r in result if r.strip()]
    result = [r.strip() for r in result if r.strip()]
    # print('\n'.join(result))
    return list(result)


def classify_website(site, text_classifier):
    """ Returns vector of class probabilities
    """
    site = extract_texts_from_html(site)
    print('Website text blocks count:', len(site))
    site = ' '.join(site)
    print('Website text length:', len(site))
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
