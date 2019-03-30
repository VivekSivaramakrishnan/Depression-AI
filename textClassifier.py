from sklearn.externals import joblib
import pickle
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
import nltk.data
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
word_tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

stop_words = set(stopwords.words('english'))


def refine_sentence(word):
    table = str.maketrans('', '', string.punctuation)
    tokenized = word_tokenizer.tokenize(word)
    cleaned_and_stemmed = [lemmatizer.lemmatize(i) for i in tokenized if i not in stop_words]
    no_punctuation = list(set([i.translate(table) for i in cleaned_and_stemmed if i.translate(table)]))
    return no_punctuation


name = 'Neural Net'
clf = joblib.load('TextClassifiers/Neural Net.pkl')

with open('topWords.pkl', 'rb') as f:
    top_words = pickle.load(f)


def textPredict(text):
    refined_text = refine_sentence(text)
    text_feature = [int(top_word in refined_text) for top_word in top_words]
    return ['DEPRESSED', 'HAPPY'][clf.predict([text_feature])[0]]


if __name__ == '__main__':
    print(textPredict(input('Enter your sentence: ')))
