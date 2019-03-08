from __future__ import print_function
from sklearn import svm
from sklearn.svm import SVC
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from features import helper
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

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
# Xtrain = preprocessing.normalize(Xtrain, norm='l1')
Xtest = np.array(Xtest).astype(np.float)
# Xtest = preprocessing.normalize(Xtest, norm='l1')
# Ytrain = np.array(Ytrain).astype(np.float)
# Ytest = np.array(Ytest).astype(np.float)



# SVM

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]}]

# scores = ['recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring="accuracy")
#     print("haha")
#     clf.fit(Xtrain, Ytrain)
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = Ytest, clf.predict(Xtest)
#     print(classification_report(y_true, y_pred))
#     print()

clf = svm.SVC( kernel='rbf', gamma=1e-3, C=100)
clf.fit(Xtrain, Ytrain)
predicted = clf.predict(Xtest)
print("SVM")
print("Accuracy: " + str(accuracy_score(Ytest, predicted)))
print("Precision: \n" + str(precision_score(Ytest, predicted, average=None)))
print("Confusion matrix: \n" + str(confusion_matrix(Ytest, predicted)))
# helper.plot_confusion_matrix(Ytest, predicted, classes=helper.getLabelList(), title='Confusion matrix for SVM')
labels = ["OQ","FD","PA","FD_FQ","FD_RQ","FD_PA","JK","NF","CQ_IR","FD_IR_PA","GG","FD_PF","PF","FQ","FD_OQ","FD_NF","CQ","RQ","CQ_FD","OQ_RQ","CQ_IR_PA","IR_PA","O","NF_OQ","FD_IR","CQ_PA","IR","PA_PF","CQ_FQ","IR_OQ","FD_FQ_NF","FQ_RQ","FQ_IR"]
cm = confusion_matrix(Ytest, predicted, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("F1 score: \n" + str(f1_score(Ytest, predicted, average='macro')))


clf2 = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=3)
clf2.fit(Xtrain, Ytrain)
predicted2 = clf2.predict(Xtest)
print("Random forest: Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted2)))
print("Accuracy: " + str(accuracy_score(Ytest, predicted2)))
print("Precision: \n" + str(precision_score(Ytest, predicted2, average=None)))
# print("Confusion matrix: \n" + str(confusion_matrix(Ytest, predicted2)))
print("F1 score: \n" + str(f1_score(Ytest, predicted2, average='macro')))

print("Confusion matrix: \n" + str(confusion_matrix(Ytest, predicted)))
# helper.plot_confusion_matrix(Ytest, predicted, classes=helper.getLabelList(), title='Confusion matrix for SVM')
labels = ["OQ","FD","PA","FD_FQ","FD_RQ","FD_PA","JK","NF","CQ_IR","FD_IR_PA","GG","FD_PF","PF","FQ","FD_OQ","FD_NF","CQ","RQ","CQ_FD","OQ_RQ","CQ_IR_PA","IR_PA","O","NF_OQ","FD_IR","CQ_PA","IR","PA_PF","CQ_FQ","IR_OQ","FD_FQ_NF","FQ_RQ","FQ_IR"]
cm = confusion_matrix(Ytest, predicted2, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# importances = clf2.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf2.estimators_],axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(Xtrain.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(Xtrain.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
# plt.xticks(range(Xtrain.shape[1]), indices)
# plt.xlim([-1, Xtrain.shape[1]])
# plt.show()


# clf3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
# clf3.fit(Xtrain, Ytrain)
# predicted3 = clf3.predict(Xtest)
# print("Logistic regression")
# print("Accuracy: " + str(accuracy_score(Ytest, predicted3)))

# # accuracy 
# # precision
# # recal 
# # f1


# clf4 = AdaBoostClassifier(n_estimators=50, learning_rate=1)
# clf4.fit(Xtrain, Ytrain)
# predicted4 = clf4.predict(Xtest)
# print("AdaBoost: Fraction of correct predictions: " + str(accuracy_score(Ytest, predicted4)))


# # from sklearn.datasets import load_iris
# # from sklearn.model_selection import train_test_split
# # iris = load_iris()
# # X, y = iris.data, iris.target
# # y[y != 1] = -1
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# # print(type(X_test))
# # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# # print(clf.score(X_test, y_test))
