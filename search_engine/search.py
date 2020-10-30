from utils.tools import cut_for_search, id2dict


class SearchEngine(object):

    def __init__(self, model, config, logger):
        # load config
        self.data_path = config['data_path']
        self.xml_path = config['xml_path']
        self.topk = config['topk']

        self.logger = logger
        self.model = model

    def search(self, query):
        query = cut_for_search(query)
        ids = self.model.get_k_nearest(query, k=self.topk)
        result = [id2dict(self.xml_path, idx) for idx in ids]
        return result

    def __enter__(self):
        self.logger.info('Start Engine...')
        return self

    def __exit__(self, type, value, trace):
        self.logger.info('Finish Engine...')
