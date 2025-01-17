import jieba
from flask import Flask, redirect, render_template, request, url_for

from search_engine.model import BM25, Word2Vec
from search_engine.search import SearchEngine
from search_engine.spider import Spider
from utils.logger import Logger
from utils.tools import load_yaml

app = Flask(__name__, static_url_path='')


@app.route("/", methods=['POST', 'GET'])
def main():
    if request.method == 'POST' and request.form.get('query'):
        query = request.form['query']
        return redirect(url_for('search', query=query))

    return render_template('index.html')


@app.route("/q/<query>", methods=['POST', 'GET'])
def search(query):
    docs = engine.search(query)
    terms = list(jieba.cut_for_search(query))
    result = highlight(docs, terms)
    return render_template('search.html', docs=result, value=query, length=len(docs))


def highlight(docs, terms):
    result = []
    for doc in docs:
        for term in terms:
            doc['abst'] = doc['abst'].replace(term, '<em><font color="red">{}</font></em>'.format(term))
            doc['title'] = doc['title'].replace(term, '<em><font color="red">{}</font></em>'.format(term))
        result.append(doc)
    return result


if __name__ == "__main__":
    config = load_yaml('settings.yaml')
    logger = Logger(config['log_path'])

    # 启动爬虫
    if config['real_time'] is True:
        spider = Spider(config, logger)

    # 选择算法
    if config['model'].lower() == 'bm25':
        model = BM25(config, logger)
    elif config['model'].lower() == 'word2vec':
        model = Word2Vec(config, logger)

    # 启动搜索引擎
    net = config['server']
    with SearchEngine(model, config, logger) as engine:
        app.run(host=net['ip_addr'], port=net['port'])
