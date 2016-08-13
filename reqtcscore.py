import nltk
from nltk.corpus import brown


class ReqTCScore:
    def __init__(self):
        self.brown_train = brown.tagged_sents(categories='news')
        self.regexp_tagger = nltk.RegexpTagger(
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
        self.unigram_tagger = nltk.UnigramTagger(self.brown_train, backoff=self.regexp_tagger)
        self.bigram_tagger = nltk.BigramTagger(self.brown_train, backoff=self.unigram_tagger)

        self.cfg = {}
        self.cfg["NNP+NNP"] = "NNP"
        self.cfg["NN+NN"] = "NNI"
        self.cfg["NNI+NN"] = "NNI"
        self.cfg["JJ+JJ"] = "JJ"
        self.cfg["JJ+NN"] = "NNI"

    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

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

    def extract(self, text):

        tokens = self.tokenize_sentence(text)
        tags = self.normalize_tags(self.bigram_tagger.tag(tokens))

        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = self.cfg.get(key, '')
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
                # if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
        return matches

    def score(self, req, tc):
        req_ext = set(self.extract(req))
        tc_ext = set(self.extract(tc))
        return len(req_ext.intersection(tc_ext)) / len(req_ext)

    def dist_score(self, reqs, tc, sort=True):
        distribution = [list(enumerate(self.score(req, tc) for tc in tc)) for req in reqs]
        return list(enumerate(distribution))

    # def dist_score_sorted(self, reqs, tc, top=5):
    #     distribution = list(enumerate(self.score(req, tc) for tc in tc) for req in reqs)
    #     return list(enumerate(distribution))

    # def dist_score_sort(self, reqs, tc, sort=True):
    #     distribution = [list(enumerate(self.score(req, tc) for tc in tc)) for req in reqs]
    #     if(sort):
    #         dist_sort = [sorted(d[1], key=lambda tup: tup[1]) for d in distribution]
    #         return list(enumerate(dist_sort))
    #     return list(enumerate(distribution))