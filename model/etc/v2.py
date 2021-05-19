#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re    
import nltk 
import string 
import warnings


# In[2]:


warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)


# In[3]:


data = pd.read_csv("../../data/tweets_combined.csv")


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


DATASET_COLUMNS = ["S_no", "Tweet", "Depressive"]
data.columns = DATASET_COLUMNS
data.head()


# In[7]:


data.drop(["S_no"], axis = 1, inplace = True)


# In[8]:


data.shape


# In[9]:


data["Depressive"].value_counts()


# In[10]:


ax = sns.barplot(x = data["Depressive"].unique(),y = data["Depressive"].value_counts());
ax.set(xlabel="Values");


# In[11]:


data.head(10)


# In[12]:


urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'
data["Processed_Tweet"] = data["Tweet"].str.replace(userPattern, "")
data["Processed_Tweet"] = data["Processed_Tweet"].str.replace(urlPattern, "")
data["Processed_Tweet"] = data["Processed_Tweet"].str.replace(r"pic.twitter.com\S+", "")
data.head(10)


# In[13]:


data["Processed_Tweet"] = data["Processed_Tweet"].str.replace("[^a-zA-Z]", " ")
data.head(10)


# In[14]:


nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


# In[15]:


data.drop(["Tweet"], axis = 1, inplace = True)


# In[16]:


data.head()


# In[17]:


def remove_stopwords(text):
    clean_text = " ".join([word for word in text.split() if word not in stopwords])
    return clean_text


data["Processed_Tweet"] = data["Processed_Tweet"].apply(lambda text: remove_stopwords(text.lower()))
data.head()


# In[18]:


data["Processed_Tweet"] = data["Processed_Tweet"].apply(lambda x: x.split())
data.head()


# In[19]:


from nltk.stem.porter import *
stemmer = PorterStemmer()
data["Processed_Tweet"] = data["Processed_Tweet"].apply(lambda x: [stemmer.stem(i) for i in x])
data.head()


# In[20]:


data["Processed_Tweet"] = data["Processed_Tweet"].apply(lambda x: ' '.join([w for w in x]))
data.head()


# In[21]:


data["Processed_Tweet"] = data["Processed_Tweet"].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
data.head()


# In[32]:


from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


# In[33]:


count_vectorizer = CountVectorizer(stop_words='english')
cv = count_vectorizer.fit_transform(data["Processed_Tweet"])
cv.shape


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(cv, data["Depressive"], test_size=.2, stratify=data["Depressive"], random_state=42)


# In[35]:


X_train.shape


# In[51]:


X_test.shape


# In[37]:


y_train.shape


# In[38]:


y_test.shape


# In[39]:


nb = MultinomialNB()
nb.fit(X_train, y_train)
prediction_nb = nb.predict(X_test)

print(accuracy_score(prediction_nb, y_test))
print(precision_score(y_test, prediction_nb, labels=[0], average='micro'))
print(recall_score(y_test, prediction_nb, labels=[0], average='micro'))
print(f1_score(y_test, prediction_nb, labels=[0], average='micro'))


# In[40]:


xgbc = XGBClassifier(max_depth=6, n_estimators=1000, nthread=3)
xgbc.fit(X_train, y_train)
prediction_xgb = xgbc.predict(X_test)

print(accuracy_score(prediction_xgb, y_test))
print(precision_score(y_test, prediction_xgb, labels=[0], average='micro'))
print(recall_score(y_test, prediction_xgb, labels=[0], average='micro'))
print(f1_score(y_test, prediction_xgb, labels=[0], average='micro'))

