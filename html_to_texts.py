#!/usr/bin/env python3

import re
import urllib.request


def extract_textx_from_html(html):
    txt = re.sub("(?s)<[^>]*>(\\s*<[^>]*>)*", r" ", site.decode()).replace('-->', '\n')
    txt = [t.strip() for t in txt if t.strip()]
    return txt


def classify_website(url)
    site = urllib.request.urlopen(url).read()
    print(len(site))
    print('\n\n\n'.join(txt.split('\n\n')))


def main():
    url = 'http://studia.elka.pw.edu.pl/EN/'
    print(classify_website(url))


if __name__ == '__main__':
    main()
