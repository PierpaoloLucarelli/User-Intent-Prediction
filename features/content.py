# returns the number of questions marks in utterance
from __future__ import division

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

