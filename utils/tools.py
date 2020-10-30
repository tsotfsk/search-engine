import pickle
import re
import xml.etree.ElementTree as ET

import jieba
import numpy as np
import requests
import yaml
from bs4 import BeautifulSoup

headers = {
    'user-agent':
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36'
}


def load_yaml(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        config = yaml.load(f)
    return config


def load_pkl(filename):
    with open(filename, 'rb') as f:
        news = pickle.load(f)
    return news


def url2soup(url):
    response = requests.get(url=url, headers=headers)
    if response.status_code != 200:
        raise AttributeError('fail...')
    response.encoding = 'utf-8'
    return BeautifulSoup(response.text, features="lxml")


def soup2dict(idx, url, soup):
    info_dict = {}
    info_dict['id'] = str(idx)
    info_dict['url'] = url
    info_dict['title'] = soup.find('title').get_text()
    info_dict['datetime'] = soup.find(attrs={"class": "time fix"}).find('span').get_text()
    raw_text = soup.find(attrs={"class": "content all-txt"}).get_text()
    info_dict['body'] = raw2cook(raw_text)
    return info_dict


def dict2xml(info_dict):
    doc = ET.Element("doc")
    ET.SubElement(doc, "id").text = info_dict['id']
    ET.SubElement(doc, "url").text = info_dict['url']
    ET.SubElement(doc, "title").text = info_dict['title']
    ET.SubElement(doc, "datetime").text = info_dict['datetime']
    ET.SubElement(doc, "body").text = info_dict['body']
    return ET.ElementTree(doc)


def id2dict(xml_path, idx):
    root = ET.parse(xml_path + f'/{idx}.xml').getroot()
    body = root.find('body').text
    abst = body[:120] + '...'
    info_dict = {
        'title': root.find('title').text,
        'abst': abst,
        'id': int(root.find('id').text),
        'url': root.find('url').text,
        'datetime': root.find('datetime').text,
    }
    return info_dict


def save_xml(xml, save_path):
    xml.write(save_path, encoding='utf-8', xml_declaration=True)


def save_news(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def raw2cook(raw_text):
    tail = '本文系观察者网独家稿件，未经授权，不得转载。'
    raw_text = re.sub(tail, '', raw_text)
    cook_strs = re.sub(r'([\r]*[\n]*)', '', raw_text).split('\t')
    cook_text = ''.join(filter(lambda x: x != '', cook_strs))
    return cook_text


def load_stop_words(pathname):
    stop_words = []
    with open(pathname, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_words.append(line.strip())
    return stop_words


def normal(array, eps=1e-15):
    if len(array.shape) > 1:
        lens = np.sqrt(np.sum(array ** 2, axis=1, keepdims=True)) + eps
    else:
        lens = np.sqrt(np.sum(array ** 2)) + eps
    return array / lens


def cut_for_search(query, stop_words=None):
    if stop_words is None:
        stop_words = []
    query = jieba.cut_for_search(query)
    return list(set(query) - set(stop_words))
