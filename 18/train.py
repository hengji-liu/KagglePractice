import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

news = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25,
                                                    random_state=33)
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

if __name__ == "__main__":
    # double underscore
    # print(SVC().get_params().keys())
    # n_jobs=-01, using all cpu cores
    gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    print(gs.best_params_, gs.best_score_)
    print(gs.score(X_test, y_test))
