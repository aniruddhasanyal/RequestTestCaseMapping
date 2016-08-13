# coding=UTF-8
import nltk, string
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
import io, re
import pandas as pd
from pprint import pprint

# This is a fast and simple noun phrase extractor (based on NLTK)
# Feel free to use it, just keep a link back to this post
# http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
# Create by Shlomi Babluki
# May, 2013

def normalize(text):
    return nltk.word_tokenize(text.lower())

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


# This is our fast Part of Speech tagger
#############################################################################
brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)
#############################################################################


# This is our semi-CFG; Extend it according to your own needs
#############################################################################
cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
#############################################################################


class NPExtractor(object):

    def __init__(self, sentence):
        self.sentence = sentence

    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged

    # Extract the main topics from the sentence
    def extract(self):

        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))

        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break

        matches = []
        for t in tags:
            if t[1] == "NNP" or t[1] == "NNI":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
        return matches

sample_file = io.open("../data/RMSRequirements.txt", 'r', encoding="iso-8859-1")
text = sample_file.read()

modules = re.findall(r'(\n{2,4}[0-9]\.\s+)(.{1,30}$)', text, re.M)
module_names = [modules[i][1] for i in range(modules.__len__())]

req_heads = re.findall(r'([0-9]\.[0-9]+\.)(.+$)+', text, re.M)
requirements = re.findall(r'([0-9]\.[0-9]+\.)((.+(\n){1})+(\n){1,4})', text, re.M)
reqs_mod_1 = [req[1] for req in requirements if int(float(req[0][:-1])) == 1]

pprint(req_heads)
print('\n\n----------------\n\n')
pprint(requirements)
print('\n\n----------------\n\n')

t_cases = pd.read_csv('../data/PO_TC.csv')
documents = list(t_cases.Description)

print(type(NPExtractor(reqs_mod_1[0]).extract()[0]))

reqs_mod_1_ext = list(' + '.join(NPExtractor(req).extract()) for req in reqs_mod_1)
docs_ext = list(' + '.join(NPExtractor(doc).extract()) for doc in documents)

pprint(reqs_mod_1_ext)
print('\n\n----------------\n\n')
pprint(docs_ext)
print('\n\n----------------\n\n')
# np_extractor = NPExtractor(sentence)
# result = np_extractor.extract()

req_tc_dist = [list(enumerate(cosine_sim(req,tc) for tc in docs_ext)) for req in reqs_mod_1_ext]
pprint(list(enumerate(req_tc_dist)))

# Main method, just run "python np_extractor.py"
def main():

    sentence = "Swayy is a beautiful new dashboard for discovering and curating online content."
    np_extractor = NPExtractor(sentence)
    result = np_extractor.extract()
    print("This sentence is about: %s" % ", ".join(result))

if __name__ == '__main__':
    main()