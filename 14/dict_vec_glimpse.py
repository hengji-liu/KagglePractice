from sklearn.feature_extraction import DictVectorizer

measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

# [[  01.   0.   0.  33.]
#  [  0.   01.   0.  12.]
#  [  0.   0.   01.  18.]]
# ['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']
