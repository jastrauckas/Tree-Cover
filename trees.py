import csv
import StringIO
import numpy 
import scipy
import sklearn
from sklearn import svm, cross_validation, tree, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
takes in the training features and the training labels
prints out the performance accuracy on a holdout set
'''
def testOnTraining(X,y):
	n = X.shape[0]
	kf = cross_validation.KFold(n, n_folds=3)

	clf = makeSVM()
	
	accuracies = []
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf.fit(X_train,y_train)
		accuracies.append( clf.score(X_test,y_test) )

	ave_accuracy = sum(accuracies)/len(accuracies)
	print "%f performance accuracy" % ave_accuracy

def train():
	print "Full training not yet implemented"

def test():
	print "Full testing not yet implemented"


def main():
	X = loadTrainingData()
	y = loadTrainingLabels()
	X2 = generateFeatures(X)
	X3 = sklearn.preprocessing.normalize(X2)
	print X3.shape
	testOnTraining(X3,y)
	return 0

if __name__ == "__main__":
	main()