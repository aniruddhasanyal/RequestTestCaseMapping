
# #################################################################################################
# The goal is to read a requirement document to extract the individual requirements and
# then try to map them to a list of available corresponding test cases. The final expectation
# from this solution is to be able to suggest a set of relevant test case for a given requirement.
# #################################################################################################

import io
import re
import pandas as pd
# import numpy as np
from nltk.corpus import wordnet
import nltk
from gensim import corpora, models, similarities
from collections import defaultdict
import string
from pprint import pprint

# ignore_words = ['']

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
stoplist = set(stopwords)

# documents = [
#     'Car Insurance',  # doc_id 0
#     'Car Insurance Coverage',  # doc_id 1
#     'Auto Insurance',  # doc_id 2
#     'Best Insurance',  # doc_id 3
#     'How much is car insurance',  # doc_id 4
#     'Best auto coverage',  # doc_id 5
#     'Auto policy',  # doc_id 6
#     'Car Policy Insurance',  # doc_id 7
# ]

# def get_wordnet_pos(pos_tag):
#     if pos_tag[1].startswith('J'):
#         return (pos_tag[0], wordnet.ADJ)
#     elif pos_tag[1].startswith('V'):
#         return (pos_tag[0], wordnet.VERB)
#     elif pos_tag[1].startswith('N'):
#         return (pos_tag[0], wordnet.NOUN)
#     elif pos_tag[1].startswith('R'):
#         return (pos_tag[0], wordnet.ADV)
#     else:
#         return (pos_tag[0], wordnet.NOUN)

def score(req, corpus, lsi, dictionary):
    vec_pos = nltk.pos_tag(nltk.word_tokenize(req.lower()))
    vec_sel = [w for (w,t) in vec_pos if (t.startswith('N')) or (t.startswith('V'))]
    vec_bow = dictionary.doc2bow(vec_sel)
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims

# --------------Extract the individual requirements from a text file--------------------
sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)
# --------------------------------------------------------------------------------------

# --------------------------Read in the test cases--------------------------------------
t_cases = pd.read_csv('../data/Book1.csv')
documents = list(t_cases.ix[:,0])
# --------------------------------------------------------------------------------------

# ------Clean up the text in the test cases and performing POS tagging----------
#    ----for Nouns & Verbs to create a LSI model to compare similarity---
texts = [[word.lower() for word in document.split()
          if word.lower() not in stoplist]
         for document in documents]

# texts = [' '.join(words) for words in texts]

# text_pos = [nltk.pos_tag(nltk.word_tokenize(rw)) for rw in texts]

# texts_select = [[w for (w,t) in row if (t.startswith('N')) or (t.startswith('V'))] for row in text_pos]
# texts_select = [[w for (w,t) in row if t.startswith('V')] for row in text_pos]

# pprint(texts)
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts_select = [[token for token in text if frequency[token] > 1]
         for text in texts]
dictionary = corpora.Dictionary(texts_select)

# doc2bow counts the number of occurences of each distinct word,
# converts the word to its integer word id and returns the result
# as a sparse vector

corpus = [dictionary.doc2bow(text) for text in texts_select]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
# --------------------------------------------------------------------------------------

req_tc_dist = [score(req[1], corpus, lsi, dictionary) for req in req_heads]

pprint(list(enumerate(req_tc_dist)))

# # ------------------Preparing the requirements for comparison---------------------------
# doc = "giraffe poop car murderer"
# vec_bow = dictionary.doc2bow(doc.lower().split())
# # --------------------------------------------------------------------------------------
#
# # -------------------------convert the query to LSI space-------------------------------
# vec_lsi = lsi[vec_bow]
# index = similarities.MatrixSimilarity(lsi[corpus])
# # --------------------------------------------------------------------------------------
#
# # -------------------perform a similarity query against the corpus----------------------
# sims = index[vec_lsi]
# sims = sorted(enumerate(sims), key=lambda item: -item[1])
#
# print(sims)