# returns the number of questions marks in utterance
from __future__ import division
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def count_qs(utterances):
	n = [];
	for u in utterances:
		if u.find('?') != -1:
			n.append(1)
		else:
			n.append(0)
	return n


def W5H1(utterances):
	# how, what, why, who, where, when
	res = []
	for utterance in utterances:
		wh_vector = [0] * 6

		wh_vector[0] = 1 if utterance.find('how') != -1 else 0
		wh_vector[1] = 1 if utterance.find('what') != -1 else 0
		wh_vector[2] = 1 if utterance.find('why') != -1 else 0
		wh_vector[3] = 1 if utterance.find('who') != -1 else 0
		wh_vector[4] = 1 if utterance.find('where') != -1 else 0
		wh_vector[5] = 1 if utterance.find('when') != -1 else 0


		res.append(list(map(str, wh_vector)))
	return res

def thanks(utterances):
	n = [];
	for u in utterances:
		if u.find('thank') != -1:
			n.append(1)
		else:
			n.append(0)
	return n

def abs_position(utterances):
	pos = [0]*len(utterances)
	for i, utterance in enumerate(utterances):
		pos[i] = i
	return pos

def norm_position(utterances):
	pos = [0]*len(utterances)
	for i, utterance in enumerate(utterances):
		pos[i] = round(i / len(utterances), 2)
	return pos	

def isUser(userTypes):
	responses = []
	for i, t in enumerate(userTypes):
		if(userTypes[i] == "User"):
			responses.append(1)
		else:
			responses.append(0)
	return responses

def vaderSentiment(utterances):
	sentiments = []
	analyzer = SentimentIntensityAnalyzer()
	for utterance in utterances:
		sent = analyzer.polarity_scores(utterance)
		s = {
			"pos": round(sent["pos"], 2),
			"neu": round(sent["neu"], 2),
			"neg": round(sent["neg"], 2),
		}
		sentiments.append(s)
	return sentiments

def duplicates(utterances):
	# how, what, why, who, where, when
	res = []
	for utterance in utterances:
		wh_vector = [0] * 2

		wh_vector[0] = 1 if utterance.find('same') != -1 else 0
		wh_vector[1] = 1 if utterance.find('similar') != -1 else 0
		res.append(list(map(str, wh_vector)))
	return res

def numberOfWords(utterances):
	lengths = []
	stop_words = set(stopwords.words('english')) 
	for utterance in utterances:
		word_tokens = word_tokenize(utterance) 
		filtered_sentence = [w for w in word_tokens if not w in stop_words] 
		lengths.append(len(filtered_sentence))
	return lengths

def numberOfUniqueWords(utterances):
	lengths = []
	stop_words = set(stopwords.words('english')) 
	for utterance in utterances:
		word_tokens = word_tokenize(utterance) 
		filtered_sentence = [w for w in word_tokens if not w in stop_words] 
		lengths.append(len(set(filtered_sentence)))
	return lengths


def exclam_mark(utterances):
	n = [];
	for u in utterances:
		if u.find('!') != -1:
			n.append(1)
		else:
			n.append(0)
	return n

def feedback(utterances):
	res = []
	for utterance in utterances:
		wh_vector = [0] * 2

		wh_vector[0] = 1 if utterance.find('did not') or utterance.find('didn\'t') != -1 else 0
		wh_vector[1] = 1 if utterance.find('does not') or utterance.find('doesn\'t') != -1 else 0

		res.append(list(map(str, wh_vector)))
	return res

# def bagOfWords(utterances):
# 	res = []


