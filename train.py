from sklearn import svm
from sklearn.metrics import accuracy_score

trainfile = "./data/features.csv"
testfile = "./data/testfeats.csv"

Xtrain = []
Ytrain = []

Xtest = []
Ytest = []

with open(trainfile) as features:
	for line in features:
		if line != '\n':
			tokens = line.strip().split(',')
			Ytrain.append(tokens[0])
			Xtrain.append(tokens[1:len(tokens)])

with open(testfile) as features:
	for line in features:
		if line != '\n':
			tokens = line.strip().split(',')
			Ytest.append(tokens[0])
			Xtest.append(tokens[1:len(tokens)])

clf = svm.SVC(gamma='scale')
clf.fit(Xtrain, Ytrain)
predicted = clf.predict(Xtest)
print("Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted)))


