# #################################################################################################
# The goal is to read a requirement document to extract the individual requirements and
# then try to map them to a list of available corresponding test cases. The final expectation
# from this solution is to be able to suggest a set of relevant test case for a given requirement.
# #################################################################################################

import io
import re
import pandas as pd
import nltk
from gensim import corpora, models, similarities
from collections import defaultdict
import string
from pprint import pprint

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
stoplist = set(stopwords)


def score(req, corpus, lsi, dictionary):
    vec_pos = nltk.pos_tag(nltk.word_tokenize(req.lower()))
    vec_sel = [w for (w, t) in vec_pos if (t.startswith('N')) or (t.startswith('V'))]
    vec_bow = dictionary.doc2bow(vec_sel)
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims


def score_mat(q_set, c_set):
    texts = [[word.lower() for word in document.split()
              if word.lower() not in stoplist]
             for document in c_set]

    texts = [' '.join(words) for words in texts]

    text_pos = [nltk.pos_tag(nltk.word_tokenize(rw)) for rw in texts]

    texts_select = [[w for (w, t) in row if (t.startswith('N')) or (t.startswith('V'))] for row in text_pos]

    frequency = defaultdict(int)
    for text in texts_select:
        for token in text:
            frequency[token] += 1
    texts_select = [[token for token in text if frequency[token] > 1]
                    for text in texts_select]
    dictionary = corpora.Dictionary(texts_select)

    corpus = [dictionary.doc2bow(text) for text in texts_select]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    req_tc_dist = [score(req, corpus, lsi, dictionary) for req in q_set]
    return list(enumerate(req_tc_dist))


# --------------Extract the individual requirements from a text file--------------------
sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)
reqs = [req[1] for req in requirements]
# --------------------------------------------------------------------------------------

# --------------------------Read in the test cases--------------------------------------
t_cases = pd.read_csv('../data/PO_TC.csv')
documents = list(t_cases.Description)
# --------------------------------------------------------------------------------------

req_tc_dist = score_mat(reqs, documents)

pprint(req_tc_dist)
