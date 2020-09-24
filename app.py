# Developer : Bhirawa Satrio Nugroho #

### Importing the libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import streamlit as st

### UI features selection
st.write("""
# Credit Scoring Prediction App

This app predict a person's creditworthiness from süddeutschen Großbank
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    laufkont = st.sidebar.slider('Laufkonto', 1, 4, 2)
    laufzeit = st.sidebar.slider('Laufzeit', 1, 72, 40)
    moral = st.sidebar.slider('Moral', 0, 4, 2)
    sparkont = st.sidebar.slider('Sparkonto', 1, 5, 3)
    data = {'laufkont':laufkont,
            'laufzeit':laufzeit,
            'moral':moral,
            'sparkont':sparkont}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

### Feature selected
st.subheader('User Input Parameters')
st.write(df)

### Features selection
dataset = pd.read_csv('kredit.csv')
X = dataset[['laufkont', 'laufzeit', 'moral','sparkont']]
y = dataset['kredit']

### Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

### Feature Scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


### Training (build logit model)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)


### Predicting single value
the_pred = classifier.predict(sc.transform(df))
the_proba = classifier.predict_proba(sc.transform(df)) * 100

### Predicting the test set results
y_pred = classifier.predict(X_test)


### prediction x reality
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.values.reshape(len(y_test),1)),1))
the_table = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.values.reshape(len(y_test),1)),1)

st.subheader('Prediction')
st.write(the_pred)


### Confusion Matrix

#real result y_true, but since we want to distingush of real results in the training set
#we called it y_true to true, y_train as training set, y_test as test set
#cm = confusion_matrix(y_test, y_pred)

#print(cm)
score = accuracy_score(y_test, y_pred)

st.subheader('Single Prediction Score')
st.write(the_proba)


st.subheader('The table')
st.write(the_table)