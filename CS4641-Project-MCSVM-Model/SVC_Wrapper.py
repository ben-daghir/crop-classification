import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import svm, datasets
from matplotlib import style

# Generalized SVC model, wrapped for our project.
class SVC_Wrapper():

    # It may be better to use LinearSVC() as the model, because this SVC apparently doesn't scale well.
    def __init__(self,model = svm.SVC(kernel="linear")):
        self.svc = model;
        return

    # train the model.
    def train(self,X_training,y_training):
        self.svc.fit(X_training,y_training)
        return

    # Returns the accuracy of the test.
    def test(self,X_testing,y_testing):
        predictions = self.svc.predict(X_testing)
        accuracy = np.sum(predictions == y_testing) / len(y_testing)
        return round(accuracy,4);

    # Got the saving code from a intro to python programming book:
    # "Hello World!" ~ By: Warren and Carter Sande
    # This stack overflow gave a simple example for the iris dataset too.
    # https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn

    # save model
    def save(self,filename):
        pickle_file = open( filename + '.pkl','wb')
        pickle.dump(self.svc, pickle_file)
        return

    # Load model
    def load(self,filename):
        pickle_file = open( filename + '.pkl','rb')
        self.svc = pickle.load(pickle_file)
        return

    # visualize the model.
    def visualize(self):
        return

def example():
    iris_data = datasets.load_iris(return_X_y=True);
    X = iris_data[0]
    y = iris_data[1]

    # Partition
    folds = 10;
    testIndexStart= int(len(y) - (len(y) / folds))
    X_training = X[:(testIndexStart - 1)]
    y_training = y[:(testIndexStart - 1)]

    X_testing = X[testIndexStart:]
    y_testing = y[testIndexStart:]

    model = svm.SVC(kernel='linear')
    SVC = SVC_Wrapper(model=model)
    SVC.train(X_training,y_training)
    accuracy = SVC.test(X_testing,y_testing)
    print( "accuracy: " + str(accuracy))
    filename = "SVC_linear_kernel"
    print("saving model as: " + str(filename) + "...")
    SVC.save(filename)

def load_example():
    iris_data = datasets.load_iris(return_X_y=True);
    X = iris_data[0]
    y = iris_data[1]

    # Partition
    folds = 10;
    testIndexStart= int(len(y) - (len(y) / folds))
    X_training = X[:(testIndexStart - 1)]
    y_training = y[:(testIndexStart - 1)]

    X_testing = X[testIndexStart:]
    y_testing = y[testIndexStart:]

    filename = "SVC_linear_kernel"
    SVC = SVC_Wrapper()
    print("loading model: " + str(filename))
    SVC.load(filename)
    print("load succeeded!")
    print("testing model...")
    accuracy = SVC.test(X_testing,y_testing)
    print( "accuracy: " + str(accuracy))
# Call example, and then load_example to demonstrate the
# ability to save a trained model!
