from gensim import corpora, models, similarities
from collections import defaultdict
import string
import nltk

class PosDist:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords.extend(string.punctuation)
        self.stopwords.append('')
        self.stoplist = set(self.stopwords)

    def _score(self, req, corpus, lsi, dictionary):
        vec_pos = nltk.pos_tag(nltk.word_tokenize(req.lower()))
        vec_sel = [w for (w, t) in vec_pos if (t.startswith('N')) or (t.startswith('V'))]
        vec_bow = dictionary.doc2bow(vec_sel)
        vec_lsi = lsi[vec_bow]
        index = similarities.MatrixSimilarity(lsi[corpus])
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        return sims

    def score_mat(self, q_set, c_set):
        texts = [[word.lower() for word in document.split()
                  if word.lower() not in self.stoplist]
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
        req_tc_dist = [self._score(req, corpus, lsi, dictionary) for req in q_set]
        return list(enumerate(req_tc_dist))
