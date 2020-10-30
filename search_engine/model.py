import math
import numpy as np
from gensim.models import Word2Vec as W2V
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.tools import cut_for_search, load_pkl, load_stop_words, normal


class BaseModel(object):
    def __init__(self, config, logger):
        # load parameters
        self.config = config
        self.logger = logger
        self.data_path = config['data_path']

        # load docs
        self.data_dict = load_pkl(self.data_path + '/texts.pkl')
        self.stop_words = load_stop_words(self.data_path + '/cn_stopwords.txt')
        docs = list(self.data_dict.values())
        self.docs = self._tokenizer(docs)

    def _tokenizer(self, docs):
        result = []
        for sen in docs:
            result.append(cut_for_search(sen, stop_words=self.stop_words))
        return result

    def sim(self, query):
        raise NotImplementedError

    def _get_k_nearest(self, query, k=5):
        raise NotImplementedError


class BM25(BaseModel):

    def __init__(self, config, logger):
        super().__init__(config, logger)
        bm2_config = config['bm25']
        self.n_docs = len(self.docs)
        self.avgdl = sum([len(doc) for doc in self.docs]) / self.n_docs
        self.k = bm2_config['k']
        self.b = bm2_config['b']

        self.d_freq = self._get_freq_in_doc()
        self.c_freq = self._get_freq_in_corpus()
        self.idf = self._get_idf()

    def _get_freq_in_doc(self):
        result = []
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
            result.append(tmp)
        return result

    def _get_freq_in_corpus(self):
        result = {}
        for tmp in self.freq:
            for k in tmp.keys():
                result[k] = result.get(k, 0) + 1
        return result

    def _get_idf(self):
        result = {}
        for word, freq in self.d_freq.items():
            result[word] = math.log(self.n_docs - freq + 0.5) - math.log(freq + 0.5)
        return result

    def _get_k_nearest(self, query, k=5):
        scores = self.sim(query)
        index = np.argsort(-scores)
        return index[:self.topk]

    def _calculate_score(self, query, idx):
        score = 0
        for word in query:
            if word not in self.d_freq[idx]:
                continue
            d = len(self.docs[idx])
            score += (self.idf[word] * self.d_freq[idx][word] * (self.k1 + 1) /
                      (self.d_freq[idx][word] + self.k *
                       (1 - self.b + self.b * d / self.avgdl)))
        return score

    def sim(self, query):
        scores = []
        for idx in range(self.n_docs):
            score = self._calculate_score(query, idx)
            scores.append(score)
        return np.array(scores)


class Word2Vec(BaseModel):

    def __init__(self, config, logger):

        super().__init__(config, logger)
        w2v_config = config['word2vec']
        self.embedding_size = w2v_config['embedding_size']
        self.window_size = w2v_config['window_size']
        self.use_weight = w2v_config['use_weight']
        self.epochs = w2v_config['epochs']
        self.use_normal = w2v_config['use_normal']

        # model
        self.model = self._build_model(embedding_size=self.embedding_size,
                                       window_size=self.window_size,
                                       epochs=self.epochs)
        self.word_dict = self._get_word_embedding()
        if self.use_weight:
            self.weight = self._tf_idf()
            self.word_dict = self._get_weight_embedding()
        self.doc_matrix = self._get_doc_embedding()

    def _tf_idf(self):
        corpus = [' '.join(sen) for sen in self.docs]
        vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        tf_idf = vectorizer.fit_transform(corpus)
        vocab = dict(sorted(vectorizer.vocabulary_.items(),
                            key=lambda x: x[1]))
        return dict(zip(vocab.keys(), tf_idf.toarray()))

    def _build_model(self, embedding_size=64, window_size=5, epochs=20):
        return W2V(self.docs,
                   size=embedding_size,
                   window=window_size,
                   min_count=1,
                   workers=4,
                   iter=epochs)

    def _get_word_embedding(self):
        return dict(zip(self.model.wv.index2word, self.model.wv.vectors))

    def _get_weight_embedding(self):
        word_weight_embedding = {}
        for word, vector in self.word_dict.items():
            word_weight_embedding[word] = vector * self.weight[word]
        return word_weight_embedding

    def _get_doc_embedding(self):
        doc_embedding = []
        for idx, sen in enumerate(self.docs):
            word_list = []
            for word in sen:
                word_list.append(self.word_dict[word])
            vec = np.sum(word_list, axis=0)
            doc_embedding.append(vec)
        doc_embedding = np.stack(doc_embedding, axis=0)

        if self.use_normal:
            return normal(doc_embedding)
        return doc_embedding

    def _get_query_embedding(self, query):
        embedding = []
        for word in query:
            embedding.append(self.word_dict[word])
        embedding = np.stack(embedding, axis=0).sum(axis=0)

        if self.use_normal:
            return normal(embedding)
        return embedding

    def sim(self, query):
        return self.doc_matrix.dot(query)

    def _get_k_nearest(self, query, k=5):
        query = self._get_query_embedding(query)
        scores = self.sim(query)
        index = np.argsort(-scores)
        return index[:k]
