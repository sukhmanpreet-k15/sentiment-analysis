import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\sentiment analysis\\IMDB_Dataset.csv')
# print(df.head(3))
df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})
print(df.head(3))