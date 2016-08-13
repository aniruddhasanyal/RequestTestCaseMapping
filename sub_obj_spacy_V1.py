import io
import re
import pandas as pd
import nltk, string
from pprint import pprint
from spacy.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from subject_object_extraction import findSVOs


remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
parser = English()

def normalize(text):
    return nltk.word_tokenize(text.lower())

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)
reqs_mod_1 = [req[1] for req in requirements if int(float(req[0][:-1]))==1]

t_cases = pd.read_csv('../data/PO_TC.csv')
documents = list(t_cases.Description)

# pprint(list(' '.join(r) for r in findSVOs(parser(reqs[0]))))

# soes = [(' '.join(line) for line in findSVOs(parser(reqs[0])))]
# print(list(' '.join(soe) for soe in soes))

req_extracts = [[' '.join(soes) for soes in findSVOs(parser(req))]for req in reqs_mod_1]
joined_extract_req = list(' '.join(ext) for ext in req_extracts)
# print(joined_extract_req)
# print(joined_extract_req.__len__())

tc_extracts = [[' '.join(soes) for soes in findSVOs(parser(req))]for req in documents]
joined_extract_tc = list(' '.join(ext) for ext in tc_extracts)
# print(joined_extract_tc)

pprint(joined_extract_req)
print('\n\n----------------\n\n')
pprint(joined_extract_tc)
print('\n\n----------------\n\n')

req_tc_dist = [list(enumerate(cosine_sim(req,tc) for tc in joined_extract_tc)) for req in joined_extract_req]
pprint(list(enumerate(req_tc_dist)))