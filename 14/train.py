from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# count
count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

# tfidf
tfidf_vec = TfidfVectorizer()
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

# count_filter
count_filter_vec = CountVectorizer(analyzer='word', stop_words='english')
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

# tfidf_filter_
tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

print('-------- count --------')

mnb_count = MultinomialNB()
mnb_count.fit(X_count_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer without filtering stopwords):',
      mnb_count.score(X_count_test, y_test))
y_count_predict = mnb_count.predict(X_count_test)
print(classification_report(y_test, y_count_predict, target_names=news.target_names))

print('-----------------------')

print()

print('-------- tfidf --------')

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes (TfidfVectorizer without filtering stopwords):',
      mnb_tfidf.score(X_tfidf_test, y_test))
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))

print('-----------------------')

print()

print('-------- count_filter --------')

mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer with filtering stopwords):',
      mnb_count_filter.score(X_count_filter_test, y_test))
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))

print('-----------------------')

print()

print('-------- tfidf_filter --------')

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes (TfidfVectorizer with filtering stopwords):',
      mnb_tfidf_filter.score(X_tfidf_filter_test, y_test))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)
print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))

print('-----------------------')
