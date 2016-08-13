######################################################################################################
# Derive the cosine similarity distribution of one list of documents with another using nltk and tfidf
######################################################################################################


import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import re
import pandas as pd
from pprint import pprint


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)


t_cases = pd.read_csv('../data/Book1.csv')
documents = list(t_cases.ix[:,0])
# pprint(documents)


req_tc_dist = [list(enumerate(cosine_sim(req[1],document) for document in documents)) for req in req_heads]
pprint(list(enumerate(req_tc_dist)))
# print(cosine_sim('a little bird', 'a little bird'))
# print(cosine_sim('a little bird', 'a little bird chirps'))
# print(cosine_sim('a little bird', 'a big dog barks'))