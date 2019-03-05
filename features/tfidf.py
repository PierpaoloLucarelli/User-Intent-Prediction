from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn.feature_selection import chi2


corpus = []
positions = []
labels = []
cosines = []
full_texts = {}

def save_tfidf(datafile):
    with open(datafile) as conversations:
        i = 0
        j = 0
		full = ''
        for line in conversations:
            if line != '\n':
                tokens = line.strip().split('\t')
                corpus.append(tokens[1])
                labels.append(tokens[0])
                positions.append(i)
                j = j + 1
				full_text = full_text + " " + tokens[1]
            else:
				full_texts[i] = full_text
                i = i + j
                j = 0

    vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()

    for count, utterance in enumerate(X):
        initial_document_pos = positions[count]
        initial_utterance = X[initial_document_pos]
        cos_sim = cosine_similarity(utterance,initial_utterance)
        cosines.append(cos_sim)

	
    return cosines

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

