---
title: "Document Classification using KNN "
date: 2019-12-15
tags: [Text processing, NLP]
---
# Document Classification

```python

from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from collections import Counter
```
*Text Preprocessing:*
Lemmatization/Stemming helps to reduce the variations among the words so that necessary words can be processed

Stop-words are removed so tat only significant text features would be captured
```python

stemmer = PorterStemmer() #stemming
lemmatizer = WordNetLemmatizer() #lemmatizing
stop_words = set(stopwords.words('english')) #removing stop words
```
Note: List Comprehensions are used which is an efficient way of iterating in python
```python
def StemTokenizer(tokens):
    return [stemmer.stem(token) for token in tokens] #List comprehensions


def Stemmer(text):
    word_tokens = word_tokenize(text.lower())
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return StemTokenizer(filtered_sentence)


def lemmaTokenizer(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def lemmatize(text):
    word_tokens = word_tokenize(text.lower())
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return lemmaTokenizer(filtered_sentence)
```

Here, if documents have to be classified according to the genre or feature we have to consider a bag of words approach.
KNN is implemented from scratch using cosine similarity as a distance measure to predict if the document is classsified accurately enough.
Standard approach is:
1) Consider the lemmatize/stemmed words and convert them to vectors using TF-TfidfVectorizer.
2) Consider training and testing dataset
3) Implement KNN to classify the documents accurately.
4) Train the model and test the model.


```python
# KNN Classifier
class KNNClassifier():
    def fit(self, dtrain, ltrain):
        self.dtrain = dtrain
        self.ltrain = ltrain
        self.vec_train = vec_tfidf.fit_transform(dtrain)

    def predict(self, data_test, k):
        vec_test = vec_tfidf.transform(data_test)
        # print(vec_test.shape)
        cos_sim = cosine_similarity(self.vec_train, vec_test)
        tcos_sim = np.transpose(cos_sim)
        neighbours = []
        for val in tcos_sim:
            distances = []
            for x in range(len(val)):
                distances.append([x, val[x]])
            distances.sort(key=operator.itemgetter(1), reverse=True)
            neighbour = []
            print(distances[1][0])
            for x in range(k):
                rowNo = distances[x][0]
                # neighbour.append([self.ltrain[rowNo], distances[x][1] ])
                neighbour.append(int(self.ltrain[rowNo]));
            # print(neighbour)
            counter = Counter(neighbour)
            neighbours.append(counter.most_common(1)[0][0])
            print(counter)
        return neighbours


def Load_text(x):
    text = []
    label = []
    counter = 1
    with open(x) as f:
        for line in f:
            if counter == 0:
                counter = counter + 1
                continue
            line.strip()
            label.append(int(line[:1]))
            text.append(line[2:])
    return text, label


def Load_test(x):
    text = []
    counter = 1
    with open(x) as f:
        for line in f:
            if counter == 0:
                counter = counter + 1
                continue
            line.strip()
            text.append(line)
    return text


# 1 load train text data
print("loading train data....")
data_train, lable_train = Load_text('train.txt')

# 2 load test data
print("loading test data....")
test_data = Load_test("test.txt")
# X_train, X_test, y_train, y_test = train_test_split(data_train, lable_train, test_size=0.2)

# 3 initialize Vectorizer
vec_tfidf = TfidfVectorizer(tokenizer=Stemmer, sublinear_tf=True, min_df=0.005, stop_words='english')
# vec_tfidf = TfidfVectorizer(tokenizer=lemmatize, stop_words = 'english' )

# 4 initialize KNNClassifier
print("initializing classifier")
classifier = KNNClassifier()
classifier.fit(data_train, lable_train)
print("prediction starting.....")
result = classifier.predict(test_data, 11)
print("writing result into file...")
np.savetxt('final1output.txt', result, delimiter='; ')
print("prediction completed....xxxxx")
```
An accuracy of 0.96 is obtained for this problem
