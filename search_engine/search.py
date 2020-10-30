import xml.etree.ElementTree as ET

from utils.tools import cut_for_search


class SearchEngine(object):

    def __init__(self, model, config, logger):
        self.data_path = config['data_path']
        self.xml_path = config['xml_path']
        self.topk = config['topk']
        self.logger = logger
        self.model = model

    def id2xml(self, idx):
        root = ET.parse(self.xml_path + f'/{idx}.xml').getroot()
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

    def search(self, query):
        sentence = cut_for_search(query)
        ids = self.model._get_k_nearest(sentence)
        result = [self.id2xml(idx) for idx in ids]
        return result

    def __enter__(self):
        self.logger.info('Start Engine...')
        return self

    def __exit__(self, type, value, trace):
        self.logger.info('Finish Engine...')
