from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk
import re
from gensim.models import word2vec

news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target


def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', '', sent.lower().strip()).split())
    return sentences


sentences = []
for x in X:
    sentences += news_to_sentences(X)

num_features = 300
min_word_count = 20
num_workers = 2
context = 5
downsampling = 1e-3

model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, \
                          min_count=min_word_count, \
                          window=context, \
                          sample=downsampling)

model.init_sims(replace=True)
model.most_similar('morning')
model.most_similar('email')
