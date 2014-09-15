import csv
import StringIO
import numpy 
import scipy
import sklearn
from sklearn import svm, cross_validation, tree, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FastICA, PCA

'''
helper function to get integer data from the csv reader
'''
def float_wrapper(reader):
    for v in reader:
        yield map(float, v)

'''
load training data from provided file and return a nump array
NOTE: the first 10 columns are quantitative data; the remaining 44 are binary
'''
def loadTrainingData():
	trainingFile = open('training_set/train_x.txt', 'rb')
	x = []
	reader = csv.reader(trainingFile, delimiter=',')
	x = list(reader)
	trainingData = numpy.array(x).astype('float')
	trainingFile.close()
	print "training data size: " + str(trainingData.shape)
	return trainingData

'''
load training labels from provided file and return a nump array
'''
def loadTrainingLabels():
	trainingFile = open('training_set/train_y.txt', 'rb')
	x = []
	reader = csv.reader(trainingFile)
	x = list(reader)
	trainingLabels = numpy.ravel(numpy.array(x).astype('float'))
	trainingFile.close()
	print "training labels size: " + str(trainingLabels.shape)
	return trainingLabels

def loadTestData():
	testFile = open('test.csv')
	reader = csv.reader(testFile, delimiter=',')
	x = list(reader)
	#headers = x[0]
	x = x[1:]
	testData = numpy.array(x).astype('float')
	testFile.close()
	return testData

'''
Do some processing on the raw training data so that we (hopefully) end up
with a lower-dimensional yet still expressive feature space
'''
def generateFeatures(X):
	# Compute ICA
	ica = FastICA(n_components=15)
	S = ica.fit_transform(X)
	X_features = S
	return X_features

'''
configure a SVM
NOTE: this is not great for a multi-class classification problem
using an SVM for multi-class classification yields a one-vs-one classifier
'''
def makeSVM(C=7):
	# value of C chosen by careful scientific evaluation
	clf = svm.SVC(C=C, kernel="rbf",tol=0.01)
	return clf

'''
configure a decision tree classifier  
kind of meta
'''
def makeTree():
	clf = tree.DecisionTreeClassifier()
	return clf

def makeForest():
	clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
	return clf

'''
relatively simple classifier, we can use this as a baseline
'''
def makeKNN():
	clf = KNeighborsClassifier()
	return clf

'''
takes in the training features and the training labels
prints out the performance accuracy on a holdout set
'''
def testOnTraining(X,y,clf):
	n = X.shape[0]
	kf = cross_validation.KFold(n, n_folds=500)
	#kf = cross_validation.KFold(n, n) # hold one out

	accuracies = []
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf.fit(X_train,y_train)
		accuracies.append( clf.score(X_test,y_test) )

	ave_accuracy = sum(accuracies)/len(accuracies)
	print "%f performance accuracy" % ave_accuracy
	return clf

'''
write out a file with the predicted labels of the test set
'''
def createPredictions(feats_train,label_train,feats_test,pid_col,clf):
	clf.fit(feats_train,label_train)
	predictions = clf.predict(feats_test)
	filename = "covertype_submission.csv"
	with open(filename,"wb") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Id","Prediction"])
		for pred,pid in zip(predictions.astype(int), pid_col.astype(int)):
			writer.writerow([pid,pred])


def train(X_train, y_train, clf):
	clf.fit(X_train, y_train)
	return clf

def main():
	X = loadTrainingData()
	y = loadTrainingLabels()
	T = loadTestData()
	id_test = T[:,0]
	X_test = T[:,1:]
	#X = generateFeatures(X)
	#X = sklearn.preprocessing.normalize(X)
	#print X.shape
	clf = makeKNN()
	testOnTraining(X, y, clf)
	train(X, y, clf)
	createPredictions(X, y, X_test, id_test, clf)
	return 0

if __name__ == "__main__":
	main()