from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn.feature_selection import chi2
import numpy as np
import pickle


def cosine_similarity(vector1, vector2):
    sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if type(sim) == np.ndarray:
        return round(sim[0], 3)
    else:
        return round(sim, 3)




def create_vectorizer(datafile,savename):
    corpus = []
    labels = []
    positions = []
    full_texts = {}
    with open(datafile) as conversations:
        i = 0
        j = 0
        full_text = ''
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
                full_text = ''
                i = i + j
                j = 0

    
    vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()
    with open(savename + 'X.pkl','wb') as f:
        pickle.dump(X, f)
    with open(savename + 'vectorizer.pkl','wb') as f:
        pickle.dump(vectorizer, f)
    with open(savename + 'positions.pkl','wb') as f:
        pickle.dump(positions, f)
    with open(savename +'full_texts.pkl','wb') as f:
        pickle.dump(full_texts, f)


def calculate_similarities_init(name):
    cosines_initial = []
    with open(name+'X.pkl','rb') as f:
        X = pickle.load(f)
    with open(name+'positions.pkl','rb') as f:
        positions = pickle.load(f)
    for count, utterance in enumerate(X):
        initial_document_pos = positions[count]
        initial_utterance = X[initial_document_pos]
        cos_sim = cosine_similarity(utterance,initial_utterance)
        cosines_initial.append(cos_sim)
    return cosines_initial



def calculate_similarities_dialog(name):
    cosines_entire = []
    with open(name+'vectorizer.pkl','rb') as f:
        vectorizer = pickle.load(f)
    with open(name+'X.pkl','rb') as f:
        X = pickle.load(f)
    with open(name+'full_texts.pkl','rb') as f:
        full_texts = pickle.load(f)
    with open(name+'positions.pkl','rb') as f:
        positions = pickle.load(f)
    for count, utterance in enumerate(X):
        i = positions[count]
        features_entire_doc = vectorizer.transform([full_texts[i]]).toarray()
        cos_simi = cosine_similarity(features_entire_doc,utterance)
        cosines_entire.append(cos_simi)
    return cosines_entire
