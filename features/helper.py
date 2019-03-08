import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def label_to_int(labels):
	newLabels = []
	with open("labels.csv") as file:
		labelList = []
		for line in file:
			labelList = line.strip().split(',')
			for i, label in enumerate(labels):
				for j, l in enumerate(labelList):
					if(label == l):
						newLabels.append(j)
						break
	return newLabels

def getLabelList():
	labels = []
	with open("labels.csv") as file:
		for line in file:
			labels = line.strip().split(',')
	return labels


def plot_confusion_matrix(y_true, y_pred, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax


def genMultiLabel(_type):
	with open("labels2.csv") as file:
		labelList = []
		for line in file:
			labelList = line.strip().split(',')
		labels = []
		with open("./data/"+_type+".tsv") as conversations:
			for line in conversations:
				tokens = line.strip().split('\t')
				if(tokens[0] != ""):
					labels.append(tokens[0])
			vectorizedLabels = [[False]*len(labelList)]*len(labels)
			for i, l in enumerate(labels):
				vectorizedLabels[i] = label2Vector(labelList, l)
		return vectorizedLabels

def label2Vector(labelList, label):
	ls = label.split("_")
	vectorLabel = [False]*len(labelList)
	for i, l in enumerate(ls):
		for j, ll in enumerate(labelList):
			if(l == ll):
				vectorLabel[j] = True
	return vectorLabel






