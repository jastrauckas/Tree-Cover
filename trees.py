import numpy 
import scipy
import sklearn
from sklearn import svm, cross_validation, tree, neighbors
import csv
import StringIO

'''
helper function to get integer data from the csv reader
'''
def int_wrapper(reader):
    for v in reader:
        yield map(int, v)

'''
load training data from provided file and return a nump array
'''
def loadTrainingData():
	trainingFile = open('training_set/train_x.txt', 'rb')
	x = []
	reader = csv.reader(trainingFile, delimiter=',')
	x = list(reader)
	trainingData = numpy.array(x).astype('int')
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
	trainingLabels = numpy.ravel(numpy.array(x).astype('int'))
	trainingFile.close()
	print "training labels size: " + str(trainingLabels.shape)
	return trainingLabels

'''
classifier options are svm or tree
'''
def makeSVM(C=7):
	# value of C chosen by careful scientific evaluation
	clf = svm.SVC(C=C, kernel="rbf",tol=0.01)
	return clf

''' 
kind of meta
'''
def makeTree():
	clf = tree.DecisionTreeClassifier()
	return clf

'''
takes in the training features and the training labels
prints out the performance accuracy on a holdout set
'''
def testOnTraining(X,y):
	n = X.shape[0]
	kf = cross_validation.KFold(n, n_folds=3)

	clf = makeTree()
	#X = sklearn.preprocessing.normalize(X)
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
	testOnTraining(X,y)
	return 0

if __name__ == "__main__":
	main()