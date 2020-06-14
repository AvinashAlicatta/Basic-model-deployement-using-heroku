import pandas as pd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
# Loading the data set

email=pd.read_csv("C://Desktop//masters//emails3.csv")
email.columns

email.Class.value_counts()
email=email.drop("Column1",axis=1)

email["Class"].value_counts()
email["Class"].value_counts(normalize=True)

import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>2:
            w.append(word)
    return (" ".join(w))


email.content = email.content.apply(cleaning_text)

# removing empty rows 
email = email.loc[email.content != " ",:]

import nltk
nltk.download('stopwords');

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  
email["st"] = email['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))

##word clouds
email_abusive=email[email.Class=='Abusive']
email_nonabusive=email[email.Class=='Non Abusive']


email_wc_a=" ".join(email_abusive.content)
email_wc_a = re.sub("[^A-Za-z" "]+"," ",email_wc_a).lower()
email_wc_a = re.sub("[0-9" "]+"," ",email_wc_a)

email_wc_a = email_wc_a.split(" ")
email_wc_a = " ".join(email_wc_a)

wordcloud_email_a = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(email_wc_a)

plt.imshow(wordcloud_email_a)


email_wc_na=" ".join(email_nonabusive.content)
email_wc_na = re.sub("[^A-Za-z" "]+"," ",email_wc_na).lower()
email_wc_na = re.sub("[0-9" "]+"," ",email_wc_na)

wordcloud_email_wc_na = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(email_wc_na)

plt.imshow(wordcloud_email_wc_na)

#bigram
def ngrams(data,n):
    text = " ".join(data)
    text1 = text.lower()
    text2 = re.sub(r'[^a-zA-Z]'," ",text1)
    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])
    words = nltk.word_tokenize(text3)
    ngram = list(nltk.ngrams(words,n))
    return ngram

ngram = ngrams(email_abusive["content"],2)
for i in range(0,len(ngram)):
    ngram[i] = "_".join(ngram[i])

Bigram_Freq = nltk.FreqDist(ngram)
bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
plt.figure(figsize = (50,25))
plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()

ngram = ngrams(email_nonabusive["content"],2)
for i in range(0,len(ngram)):
    ngram[i] = "_".join(ngram[i])
    

Bigram_Freq = nltk.FreqDist(ngram)
bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
plt.figure(figsize = (50,25))
plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()
