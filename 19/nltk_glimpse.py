from sklearn.feature_extraction.text import CountVectorizer
import nltk

sent1 = 'The cat is walking in the room'
sent2 = 'A dog was running across the kitchen'

count_vec = CountVectorizer()
sentence = [sent1, sent2]

print(count_vec.fit_transform(sentence).toarray())
# [[0 1 0 1 1 0 1 0 2 1 0]
#  [1 0 1 0 0 1 0 1 1 0 1]]

print(count_vec.get_feature_names())
# ['across', 'cat', 'dog', 'in', 'is', 'kitchen', 'room', 'running', 'the', 'walking', 'was']

# tokenization
tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)

tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)

vocab_1 = sorted(set(tokens_1))
print(vocab_1)

vocab_2 = sorted(set(tokens_2))
print(vocab_2)

# stemming
stemmer = nltk.stem.PorterStemmer()

stem_1 = {stemmer.stem(t) for t in tokens_1}
print(stem_1)

stem_2 = {stemmer.stem(t) for t in tokens_2}
print(stem_2)

# part of speach tagging
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print(pos_tag_1)

pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print(pos_tag_1)
