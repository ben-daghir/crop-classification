import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from SVC_Wrapper import SVC_Wrapper
from sklearn import model_selection, svm, datasets
from matplotlib import style

# Should return a matrix of two dimensions.
def get_training_data():
    # Iris data. :)
    iris_data = datasets.load_iris(return_X_y=True)
    X = iris_data[0]
    y = iris_data[1]

     # Partition
    folds = 10;
    testIndexStart= int(len(y) - (len(y) / folds))
    X_training = X[:(testIndexStart - 1)]
    y_training = y[:(testIndexStart - 1)]

    X_testing = X[testIndexStart:]
    y_testing = y[testIndexStart:]

    return X_training, y_training

# Should return a matrix of two dimensions.
def get_testing_data():
    # Iris data. :)
    iris_data = datasets.load_iris(return_X_y=True)
    X = iris_data[0]
    y = iris_data[1]

     # Partition
    folds = 8;
    testIndexStart= int(len(y) - (len(y) / folds))
    X_training = X[:(testIndexStart - 1)]
    y_training = y[:(testIndexStart - 1)]

    X_testing = X[testIndexStart:]
    y_testing = y[testIndexStart:]

    return X_testing, y_testing

def MAIN():

    # Get Data
    X_training, y_training = get_training_data()
    X_testing, y_testing = get_testing_data()

    # Grid search for SVC hyperparameters.
    print("Fitting the classifier to the training set...")
    t0 = time.time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    model = svm.SVC(kernel='rbf', class_weight='balanced')
    SVC_ensemble = model_selection.GridSearchCV(model, param_grid)
    Ensemble_wrapped = SVC_Wrapper(model = SVC_ensemble)
    Ensemble_wrapped.train(X_training,y_training)
    print("done in %0.3fs" % (time.time() - t0))
    print("Best estimator found by grid search:")
    bestSVC = SVC_ensemble.best_estimator_
    print(SVC_ensemble.best_estimator_)

    # Test results off the best model.
    print("Testing best SVC...")
    best_Wrapper = SVC_Wrapper(model=bestSVC)
    score = best_Wrapper.test(X_testing,y_testing)
    print("accuracy: " + str(score))

    print("Saving model...")
    best_Wrapper.save("BestSVC_output")
    return

MAIN()
