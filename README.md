# EX NO: 15

# Date

# Email-spam-classifier

# Aim:
    The aim of an email spam classifier is to automatically classify incoming emails as either spam or non-spam (ham) based on their content and other relevant features. 
    
# Algorithm:

Step 1: Import Libraries: Start by importing the necessary libraries such as numpy, pandas, and sklearn.

Step 2: Load the Dataset: If you have a labeled dataset of emails (spam and non-spam), load it using pandas or any other preferred method. If not, you may need to collect and label a dataset manually.

Step 3: Data Preprocessing: Perform necessary preprocessing steps on the email data, such as removing any irrelevant information (e.g., email addresses, timestamps), handling missing values, and normalizing the text (e.g., converting to lowercase, removing punctuation, stemming or lemmatization).

Step 4: Feature Extraction: Convert the textual content of each email into numerical features that machine learning algorithms can understand. Some common techniques for feature extraction in text classification include bag-of-words, TF-IDF, and word embeddings like Word2Vec or GloVe.


# Code:

```

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import kaggle
from zipfile import ZipFile
from os import path
if path.exists(".\emails.csv"):
    pass
else:
    with ZipFile("email-spam-classification.zip") as f:
        print(f.extractall())
df = pd.read_csv("emails.csv")
df
df.shape
df = df.iloc[:,0:2]
df.isnull().sum()
df.dropna(inplace = True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.info()
df["spam"] = df['spam'].apply(lambda x: "text_result" if len(x) > 1 else x) 
df = df[df["spam"] != "text_result"]
df['spam'] = df['spam'].astype('int')
df.info()
df.iloc[15].text
df['text'] = df['text'].apply(lambda x:x.lower())
df
df["text"][0]
def rem_special_chars(text):
    new_text = ""
    for i in text:
        if i.isalnum() or i == " ":
            new_text += i
    return new_text.strip()
df['text'] = df['text'].apply(rem_special_chars)
df
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',max_features=10000)
X = cv.fit_transform(df['text']).toarray()
y = df["spam"].values
y
array([1, 1, 1, ..., 0, 0, 0])
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
len(cv.get_feature_names())
len(cv.get_stop_words())
import pickle
pickle.dump(cv,open('model/cv.pkl','wb'))
pickle.dump(clf,open('model/clf.pkl','wb'))

```

# Output:




