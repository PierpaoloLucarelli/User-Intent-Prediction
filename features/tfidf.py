from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn.feature_selection import chi2
import pickle 

def save_tfidf(datafile, savename):

    corpus = []
    positions = []
    labels = []
    cosines = []
    print("one call")
    print(datafile)
    with open(datafile) as conversations:
        i = 0
        j = 0
        for line in conversations:
            if line != '\n':
                tokens = line.strip().split('\t')
                corpus.append(tokens[1])
                labels.append(tokens[0])
                positions.append(i)
                j = j + 1
            else:
                i = i + j
                j = 0

    vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()
    for count, utterance in enumerate(X):
        initial_document_pos = positions[count]
        initial_utterance = X[initial_document_pos]
        cos_sim = cosine_similarity(utterance,initial_utterance)
        cosines.append(cos_sim)
    with open(savename + '.pkl','wb') as f:
        pickle.dump(cosines, f)
    return cosines

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return round(dot_product/magnitude, 3)

