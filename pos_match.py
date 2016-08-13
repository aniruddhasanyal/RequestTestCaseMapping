import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
from nltk.corpus import wordnet
import string

from nltk.tokenize import word_tokenize


class PosMatch:
    def __init__(self):
        # Get default English stopwords and extend with punctuation
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords.extend(string.punctuation)
        self.stopwords.append('')
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        # Create tokenizer and stemmer
        # self.tokenizer = nltk.tokenize.punkt.PunktWordTokenizer()


    def get_wordnet_pos(pos_tag):
        if pos_tag[1].startswith('J'):
            return (pos_tag[0], wordnet.ADJ)
        elif pos_tag[1].startswith('V'):
            return (pos_tag[0], wordnet.VERB)
        elif pos_tag[1].startswith('N'):
            return (pos_tag[0], wordnet.NOUN)
        elif pos_tag[1].startswith('R'):
            return (pos_tag[0], wordnet.ADV)
        else:
            return (pos_tag[0], wordnet.NOUN)

    def pos_match(self, a, b, threshold=0.5):
        """Check if a and b are matches."""
        # pos_a = map(self.get_wordnet_pos, nltk.pos_tag(word_tokenize(a)))
        # pos_b = map(self.get_wordnet_pos, nltk.pos_tag(word_tokenize(b)))
        pos_a = [self.get_wordnet_pos(token) for token in nltk.pos_tag(word_tokenize(a))]
        pos_b = [self.get_wordnet_pos(token) for token in nltk.pos_tag(word_tokenize(b))]
        lemmae_a = [self.lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_a \
                    if pos == wordnet.NOUN and token.lower().strip(string.punctuation) not in self.stopwords]
        lemmae_b = [self.lemmatizer.lemmatize(token.lower().strip(string.punctuation), pos) for token, pos in pos_b \
                    if pos == wordnet.NOUN and token.lower().strip(string.punctuation) not in self.stopwords]

        # Calculate Jaccard similarity
        ratio = len(set(lemmae_a).intersection(lemmae_b)) / float(len(set(lemmae_a).union(lemmae_b)))
        return ratio
        # if ratio >= threshold: return ratio
        # return (ratio >= threshold)
