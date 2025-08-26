#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re #regular expressions
import nltk #natural language tool kit


# In[2]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer #used to reduce words to the root forms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


# Download Stopwords
nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# ### Preprocessing data

# In[4]:


# Load dataset
df = pd.read_csv("Downloads/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df.head()


# In[5]:


def preprocess_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem words
    return " ".join(words)


# In[6]:


df["cleaned_message"] = df["message"].apply(preprocess_text)
print(df.head())


# ### Training an ML Model

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer #converts text to numerical form
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[8]:


vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]


# In[9]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# ### Evaluating Model Perfomance

# In[11]:


y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))


# In[12]:


def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"


# In[13]:


email = "Congratulations! You've won a free iPhone. Click here to claim now."
print(f"Email: {email}\nPrediction: {predict_email(email)}")

