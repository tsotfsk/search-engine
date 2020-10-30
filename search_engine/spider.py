import time

from tqdm import tqdm
from utils.tools import dict2xml, save_news, save_xml, soup2dict, url2soup


class Spider(object):
    def __init__(self, config, logger):
        self.root = config['root']
        self.catalogue = config['catalogue']
        self.num_pages = config['num_pages']
        self.wait_time = config['wait_time']
        self.xml_path = config['xml_path']
        self.data_path = config['data_path']

        self.logger = logger

        self.page_lists = self._get_page_lists()
        self.news_lists = self._get_news_lists()
        self.xmls, self.texts = self._crawl_news()
        self._save_data()

    def _get_page_lists(self):
        page_lists = []
        for i in range(1, self.num_pages + 1):
            page_lists.append('/'.join(
                [self.root, self.catalogue, f'list_{i}.shtml']))
        return page_lists

    def _get_news_lists(self):
        news_lists = []
        for url in self.page_lists:
            try:
                soup = url2soup(url)
            except ValueError:
                continue
            for title in soup.find_all('h4'):
                suffix = title.find_all('a')[0]['href']
                news_lists.append(''.join([self.root, suffix]))
        return news_lists

    def _crawl_news(self):
        idx = 0
        xmls, texts = [], []
        self.logger.info('Start crawl news...')
        for url in tqdm(self.news_lists, total=len(self.news_lists)):
            try:
                xml, text = self._process_news(idx, url)
            except AttributeError:
                continue
            idx += 1
            xmls.append(xml)
            texts.append(text)
            time.sleep(self.wait_time)
        self.logger.info(f'Crawling news done... {idx}/{len(self.news_lists)}')
        return xmls, texts

    def _process_news(self, idx, url):
        soup = url2soup(url)
        info_dict = soup2dict(idx, url, soup)
        xml = dict2xml(info_dict)
        return xml, info_dict['body']

    def _save_data(self):
        for idx, xml in enumerate(self.xmls):
            save_xml(xml, self.xml_path + "/{}.xml".format(idx))

        texts = dict(enumerate(self.texts))
        save_news(texts, self.data_path + "/texts.pkl")
