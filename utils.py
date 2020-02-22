# Function to return accuracies for SVM and online passive-aggresive algorithms
from pa import PA
from paKernel import KPA
from paActive import AL
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

def test_on_data(X,y,classifiers):
	"""

	Prints accuracies on classifiers

	classifiers: A dictionary of classifiers
	if SVM is true, supply kernel, gamma, degree
	"""

	#data = np.array(data)
	#X = data[:,0:data.shape[1]-2]
	#y = data[:,-1]
	#y = y.reshape((y.shape[0]))
	#y = LabelBinarizer(neg_label=-1,pos_label=1).fit_transform(y)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 42)

	accuracies = []
	for clf_name,clf in classifiers.items():
		start = time.time()
		clf = clf
		clf.fit(X_train,y_train)
		clf_time = time.time()-start
		if clf_name is 'SVM':
			clf_accuracy = round(np.mean(y_test == clf.predict(X_test)) * 100,2)
			accuracies.append(clf_accuracy)
		else:
			clf_accuracy = round(np.mean(y_test == clf.predict_all(X_test)) * 100,2)   
			accuracies.append(clf_accuracy)     
		print("'%s'( time taken: "%clf_name, clf_time, ", accuracy:  ", clf_accuracy, "%)")

# Calculate accuracy
def get_accuracy(y_pred,y):
	acc = 100*(y_pred==y).mean()
	print('Accuracy is:',acc)

	return acc