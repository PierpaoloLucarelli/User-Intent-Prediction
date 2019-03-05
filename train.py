from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# trainfile = "./data/train_feat.csv"
# testfile = "./data/test_feat.csv"

testfile = "./data/test_feat.csv"
trainfile = "./data/train_feat.csv"

Xtrain = []
Ytrain = []

Xtest = []
Ytest = []

le = preprocessing.LabelEncoder()

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


# le.fit(Ytrain)
# Ytrain = list(le.transform(Ytrain))
# # Ytest = list(le.transform(Ytest))
# print(Ytest)


clf = svm.SVC(gamma='scale')
clf.fit(Xtrain, Ytrain)
predicted = clf.predict(Xtest)
print("SVM: Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted)))

clf2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2)
clf2.fit(Xtrain, Ytrain)
predicted2 = clf2.predict(Xtest)
print("Random forest: Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted2)))


clf3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf3.fit(Xtrain, Ytrain)
predicted3 = clf3.predict(Xtest)
print("Logistic regression: Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted3)))


clf4 = AdaBoostClassifier(n_estimators=50, learning_rate=1)
clf4.fit(Xtrain, Ytrain)
predicted4 = clf4.predict(Xtest)
print("AdaBoost: Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted4)))




