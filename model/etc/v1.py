"""
            *** Section 1 ***
            Import Libraries & Dataset Preparation
"""

import re    # for regular expressions
import nltk  # for text manipulation
import string  # for text manipulation
import warnings
import numpy as np
import pandas as pd  # for data manipulation
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")  # ignore warnings

# Read the dataset and covert it to dataframe
data = pd.read_csv("/content/drive/MyDrive/training.1600000.processed.noemoticon.csv", encoding='latin-1')
data.head()

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]
data.columns = DATASET_COLUMNS
data.head()

# Drop unnecessary columns
data.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)
data.head()

positive_data = data[data.target == 4].iloc[:25000, :]
print(positive_data.shape)
negative_data = data[data.target == 0].iloc[:1000, :]
print(negative_data.shape)

data = pd.concat([positive_data, negative_data], axis=0)
print(data.shape)
data.head()


"""
            *** Section - 2 ***
                Data Cleaning

"""
# Removing Twitter Handles (@user)
data['Clean_TweetText'] = data['TweetText'].str.replace("@", "")
data.head()

# Removing links
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace(r"http\S+", "")
data.head()

# Removing Punctuations, Numbers, and Special Characters
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace("[^a-zA-Z]", " ")
data.head()

# Remove stop words
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    clean_text = ' '.join([word for word in text.split() if word not in stopwords])
    return clean_text


data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda text: remove_stopwords(text.lower()))
data.head()


# Text Tokenization and Normalization
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: x.split())
data.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: [stemmer.stem(i) for i in x])
data.head()


# Now let’s stitch these tokens back together
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x]))
data.head()

# Removing small words
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
data.head()


"""
            *** Section - 3 ***
            Build and Train Model

"""
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

count_vectorizer = CountVectorizer(stop_words='english')
cv = count_vectorizer.fit_transform(data['Clean_TweetText'])
cv.shape

X_train, X_test, y_train, y_test = train_test_split(cv, data['target'], test_size=.2, stratify=data['target'], random_state=42)


# XGBoost Classification
xgbc = XGBClassifier(max_depth=6, n_estimators=1000, nthread=3)
xgbc.fit(X_train, y_train)
prediction_xgb = xgbc.predict(X_test)

print(accuracy_score(prediction_xgb, y_test))
print(precision_score(y_test, prediction_xgb, labels=[0], average='micro'))
print(recall_score(y_test, prediction_xgb, labels=[0], average='micro'))
print(f1_score(y_test, prediction_xgb, labels=[0], average='micro'))


# Naive Bayes Classification
nb = MultinomialNB()
nb.fit(X_train, y_train)
prediction_nb = nb.predict(X_test)

print(accuracy_score(prediction_nb, y_test))
print(precision_score(y_test, prediction_nb, labels=[0], average='micro'))
print(recall_score(y_test, prediction_nb, labels=[0], average='micro'))
print(f1_score(y_test, prediction_nb, labels=[0], average='micro'))
