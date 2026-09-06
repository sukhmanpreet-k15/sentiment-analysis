import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# Load the dataset
df = pd.read_csv('C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\sentiment analysis\\IMDB_Dataset.csv')
# print(df.head(3))
df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})
# print(df.head(3))
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

df['review'] = df['review'].apply(clean_html)
def to_lowercase(text):
    return text.lower()

df['review'] = df['review'].apply(to_lowercase)
def remove_special(text):
    x = ''                      # start with an empty result string

    for i in text:               # go through the text one character at a time
        if i.isalnum():          # is this character a letter or number?
            x = x + i             # if yes, keep it — add it to result
        else:
            x = x + ' '           # if no (it's a special character), add a space instead

    return x
df['review'] = df['review'].apply(remove_special)
                             #remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    x=[]
    for i in text.split():
        if i not in stop_words:
            x.append(i)

    y=x[:]
    x.clear()
    return y
df['review'] = df['review'].apply(remove_stopwords)
from nltk.stem import PorterStemmer
ps = PorterStemmer()
def stem_words(text):
    y=[]
    for i in text:
        y.append(ps.stem(i))
    z=y[:] 
    y.clear()
    return z   
df['review'] = df['review'].apply(stem_words)
def join_words(text):
    return ' '.join(text)
df['review'] = df['review'].apply(join_words)
x=df.iloc[:,0:1].values
cv=CountVectorizer()
X=cv.fit_transform(df['review'])
y = df['sentiment'].astype(int)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model1=GaussianNB()
# model1.fit(x_train, y_train)
model2=MultinomialNB()
model2.fit(x_train, y_train)
model3=BernoulliNB()    
model3.fit(x_train, y_train)
# y_pred1=model1.predict(x_test)
y_pred2=model2.predict(x_test)
y_pred3=model3.predict(x_test)
# print("Accuracy of GaussianNB:", accuracy_score(y_test, y_pred1))
print("Accuracy of MultinomialNB:", accuracy_score(y_test, y_pred2))
print("Accuracy of BernoulliNB:", accuracy_score(y_test, y_pred3))
