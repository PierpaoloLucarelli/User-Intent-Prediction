import numpy as np
from features import helper
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

testfile = "./data/test_feat.csv"
trainfile = "./data/train_feat.csv"

Xtrain = []
Ytrain = []

Xtest = []
Ytest = []


with open(trainfile) as features:
	for line in features:
		if line != '\n':
			tokens = line.strip().split(',')
			Ytrain.append(tokens[0])
			tk = list(map(float, tokens[1:len(tokens)]))
			Xtrain.append(tk)

with open(testfile) as features:
	for line in features:
		if line != '\n':
			tokens = line.strip().split(',')
			Ytest.append(tokens[0])
			tk = list(map(float, tokens[1:len(tokens)]))
			Xtest.append(tk)


Xtrain = np.array(Xtrain).astype(np.float)
Xtest = np.array(Xtest).astype(np.float)

Ytrain = helper.genMultiLabel("train")
Ytest  = helper.genMultiLabel("test")

Ytrain = np.array(Ytrain).astype(np.bool)
Ytest = np.array(Ytest).astype(np.bool)


# base_svc = LogisticRegression(solver='lbfgs')
base_svc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=3)
# base_svc = SVC( kernel='rbf', gamma=1e-3, C=100, verbose=True)
ovr = OneVsRestClassifier(base_svc)
ovr.fit(Xtrain, Ytrain)

predicted = ovr.predict(Xtest)
# print(Ytest)
# print(predicted)
print("Accuracy: " + str(accuracy_score(Ytest, predicted)))

chains = [ClassifierChain(base_svc, order='random', random_state=i) for i in range(10)]
for chain in chains:
    chain.fit(Xtrain, Ytrain)

Y_pred_chains = np.array([chain.predict(Xtest) for chain in chains])

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
# print("Accuracy: " + str(accuracy_score(Ytest, Y_pred_ensemble)))
# print(Y_pred_ensemble)
ensemble_jaccard_score = jaccard_similarity_score(Ytest,Y_pred_ensemble >= .5)

print("Jaccard: " + str(ensemble_jaccard_score))
# binY = np.where(Y_pred_ensemble>0.5, 1, 0)
# print(binY)
print("Accuracy: " + str(average_precision_score(Ytest, Y_pred_ensemble)))

