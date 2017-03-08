# scikit-learn practice

## Basic Model

**Classification**

|No.|application|model|other|
|---|-----------|-----|-----|
|01.|Breast cancer prediction|Logistic Regression VS. <br/> SGD Classifier|StandardScaler|
|02.|Digit recognition|LinearSVC (support vector classifier)|StandardScaler|
|03.|News classification|Multinomial Naive Bayes|CountVectorizer|
|04.|Iris|K-Nearest-Neighbour Classifier|StandardScalar|
|05.|Titanic|Decision Tree Classifier|DictVectorizer <br/> One-hot encoding|
|06.|Titanic|Decision Tree Classifier VS. <br/> Random Forest Classifier VS. <br/> Gradient Boosting Classifier|DictVectorizer <br/> One-hot encoding|

**Regression**

|No.|application|model|other|
|---|-----------|-----|-----|
|07.|Boston housing price|Linear Regression VS. <br/> SGDRegressor|StandardScaler <br/> R2 score, Mean Squared Error, Mean Absolute Error|
|08.|Boston housing price|Support Vector Regression|linear/poly/radial basis kernel|
|09.|Boston housing price|K Neighbour Regression|unifrom/distance-weighted|
|10.|Boston housing price|Decision Tree Regression||
|11.|Boston housing price|RandomForestRegressor VS. <br/> ExtraTreesRegressor VS. <br/> GradientBoostingRegressor| feature importance of ExtraTreesRegressor|

**Unsupervised**

|No.|application|model|other|
|---|-----------|-----|-----|
|12.|Digit recognition|K-means|Adjusted Random Index, <br/> Silhouette, <br/> Elbow Method, <br/> Matplotlib|
|13.|Digit recognition|Principal Component Analysis|Matplotlib|

## Techniques

**Feature Engineering**

|No.|application|model|other|
|---|-----------|-----|-----|
|14.|News classification|Multimonial Naive Bayes|DictVectorizer <br/> CountVectorizer VS. <br/> TfidfVectorizer VS. <br/> both with stopwords filtering|
|15.|Titanic|Decision Tree Classifier| feature_selection by percentile|

**Regularization**

|No.|application|model|other|
|---|-----------|-----|-----|
|16.|Pizza price|Linear Regression|Polynominal Features, degree =2/=4|
|17.|Pizza price|Linear Regression VS. <br/> Lasso VS. <br/> Ridge| L1, L2 penalty|

**Hyper-parameter search**

|No.|application|model|other|
|---|-----------|-----|-----|
|18.|News classification|SVC|Pipeline <br/> Parallel Grid Search|

**Packages**

|No.|application|model|other|
|---|-----------|-----|-----|
|19.|News classification||Word2Vec <br/> NLTK|
|20.|News classification|Random Forrest Classifier VS. <br/> XGB Classifier||
|21.|Breast cancer prediction <br/> Boston housing price|Tensor Flow and its sklearn style api|CODE USING API DOES NOT WORK|

## Practice

|No.|application|model|other|
|---|-----------|-----|-----|
|22.|Titanic|xgbc<br/>rfc|cv|
