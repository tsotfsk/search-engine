import jieba
from flask import Flask, redirect, render_template, request, url_for

from search_engine.model import Word2Vec, BM25
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
    print()
    for doc in docs:
        content = doc.get('title')
        for term in terms:
            content = content.replace(term, '<em><font color="red">{}</font></em>'.format(term))
        result.append(doc)
    return result


if __name__ == "__main__":
    config = load_yaml('settings.yaml')
    logger = Logger('./ir&ie.log')

    # 启动爬虫
    if config['real_time'] is True:
        spider = Spider(config, logger)

    # 选择算法
    if config['model'].lower() == 'bm25':
        model = Word2Vec(config, logger)
    elif config['model'].lower() == 'word2vec':
        model = BM25(config, logger)

    # 启动搜索引擎
    net = config['server']
    with SearchEngine(model, config, logger) as engine:
        app.run(host=net['ip_addr'], port=net['port'])
