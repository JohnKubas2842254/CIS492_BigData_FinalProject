from threading import Thread
import json
import re
from html2text import html2text as htt # type: ignore
import wikitextparser as wtp #type: ignore

def dewiki(text):
    text = wtp.parse(text).plain_text()  # xml text --> tree --> plain text 
    text = htt(text)  # remove any HTML tags
    text = text.replace('\\n',' ')  # replace newlines
    text = re.sub('\s+', ' ', text)  # replace excess whitespace
    return text


def analyze_chunk(text):
    try:
        if '<redirect title="' in text:  # this is not the main document
            return None
        if '(disambiguation)' in text:  # this is not an document
            return None
        else:
            title = text.split('<title>')[1].split('</title>')[0]
            title = htt(title)
            if ':' in title:  # most documents with : in them are not documents we care about
                return None
        serial = text.split('<id>')[1].split('</id>')[0]
        content = text.split('</text')[0].split('<text')[1].split('>', maxsplit=1)[1]
        content = dewiki(content)
        return {'title': title.strip(), 'text': content.strip(), 'id': serial.strip()}
    except Exception as oops:
        print(oops)
        return None


def save_document(document, savedir):
    doc = analyze_chunk(document)
    if doc:
        print('SAVING:', doc['title'])
        filename = doc['title'] + '.json'
        with open(savedir + filename, 'w', encoding='utf-8') as outfile:
            json.dump(doc, outfile, sort_keys=True, indent=1, ensure_ascii=False)


def process_file_text(filename, savedir):
    document = ''
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            if '<page>' in line:
                document = ''
            elif '</page>' in line:  # end of document
                Thread(target=save_document, args=(document, savedir)).start()
            else:
                document += line                