from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
df= pd.read_csv("emails3.csv", encoding="latin-1")
df=df.drop(['Column1'],axis=1)

df['label'] = df['class'].map({'Abusive': 0, 'Non Abusive': 1})
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) 

pickle.dump(cv, open('tranform.pkl', 'wb'))
    
    
#	from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#	#Naive Bayes Classifier
#	from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
    filename = 'nlp_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    
