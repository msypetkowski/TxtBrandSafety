import re
from bs4 import BeautifulSoup


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