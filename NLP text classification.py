#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


# Read our dataset using read_csv()
bbc_text = pd.read_csv(r"bbc-text.txt")
bbc_text=bbc_text.rename(columns = {'text': 'News_Headline'}, inplace = False)
bbc_text.head()


# In[12]:


bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
bbc_text.category.unique()


# In[13]:


from sklearn.model_selection import train_test_split
X = bbc_text.News_Headline
y = bbc_text.category
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)


# In[14]:


vector = CountVectorizer(stop_words = 'english',lowercase=False)
# fit the vectorizer on the training data
vector.fit(X_train)
vector.vocabulary_
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)


# In[15]:


from sklearn.naive_bayes import MultinomialNB
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(naivebayes.predict(X_test_transformed), y_test))


# In[17]:


headline1 = ['Portugal crash out of FIFA World Cup 2022, Ronaldo in tears']
vec = vector.transform(headline1).toarray()
print('Headline:', headline1)
print(str(list(naivebayes.predict(vec))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
#naivebayes.predict(vec)[0]


# In[18]:


headline1 = ['There will be recession throughout the world as predicted by world bank']
vec = vector.transform(headline1).toarray()
print('Headline:', headline1)
print(str(list(naivebayes.predict(vec))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))


# In[19]:


#to save the model
import pickle

saved_model = pickle.dumps(naivebayes)


# In[20]:


#load saved model
s = pickle.loads(saved_model)


# In[21]:


headline1 = ['There will be recession throughout the world as predicted by world bank']
vec = vector.transform(headline1).toarray()

s.predict(vec)[0]


# In[ ]:




