import numpy 
import scipy
import sklearn
from sklearn import svm, cross_validation, tree, neighbors
import csv
import StringIO

def int_wrapper(reader):
    for v in reader:
        yield map(int, v)

def loadTraining():
	trainingFile = open('training_set/train_x.txt', 'rb')
	x = []
	reader = csv.reader(trainingFile, delimiter=',')
	x = list(reader)
	trainingData = numpy.array(x).astype('int')
	trainingFile.close()
	print trainingData.shape





def main():
	loadTraining()
	return 0

if __name__ == "__main__":
	main()