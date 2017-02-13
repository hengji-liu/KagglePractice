from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')
# print(len(news.data))
# print(news.data[0])

# 18846
# From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>
# Subject: Pens fans reactions
# Organization: Post Office, Carnegie Mellon, Pittsburgh, PA
# Lines: 12
# NNTP-Posting-Host: po4.andrew.cmu.edu
#
#
#
# I am sure some bashers of Pens fans are pretty confused about the lack
# of any kind of posts about the recent Pens massacre of the Devils. Actually,
# I am  bit puzzled too and a bit relieved. However, I am going to put an end
# to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
# are killing those Devils worse than I thought. Jagr just showed you why
# he is much better than his regular season stats. He is also a lot
# fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
# fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
# regular season game.          PENS RULE!!!

X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)

print('The accuracy of Naive Bayes Classification is ', mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))
