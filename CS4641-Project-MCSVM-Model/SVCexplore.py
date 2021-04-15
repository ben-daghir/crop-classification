import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from matplotlib import style

# First exploration done with:
# https://www.youtube.com/watch?v=81ZGOib7DTk
# This is a direct follow along.
def firstExploration():
    style.use("ggplot")

    x = [1,5,1.5,8,1,9]
    y = [2,8,1.8,8,0.5,11]

    plt.scatter(x,y)
    plt.show()

    X = np.array([
        [1,2],
        [5,8],
        [1.5,1.8],
        [8,8],
        [1,0.6],
        [9,11]
    ])

    y = [0,1,0,1,0,1]

    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(X,y)

    w = clf.coef_[0]
    print(w)

    # not sure what this means. :| - sorry.
    a = -w[0] / w[1]
    xx = np.linspace(0,12)
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx,yy,'k-', label = "non weighted divide")

    plt.scatter(X[:,0],X[:,1], c = y)
    plt.legend()
    plt.show()

# second exploration done with
# https://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py
# Code follows this example, but...
# https://scikit-learn.org/stable/tutorial/basic/tutorial.html
# I am also using information from here.
def secondExploration():
    iris_data = datasets.load_iris(return_X_y=True);
    X = iris_data[0] # Modified based on suggestion.
    X = X[:, :2]
    y = iris_data[1]

    # Note to Self: Testing data set may be dependent on position in
    # the array. Not sure how Scikit learn handles this within their kernels.
    folds = 10;
    testIndexStart= int(len(y) - (len(y) / folds))
    X_training = X[:(testIndexStart - 1)]
    y_training = y[:(testIndexStart - 1)]

    X_testing = X[testIndexStart:]
    y_testing = y[testIndexStart:]
    # Just me testing scatter plots.
    # plt.scatter(X_training[:,0],X_training[:,1], c = y_training)
    # plt.show()

    # Options: linear, poly, rbf <- not sure what that last one is.
    # You can also pass a custom kernel. Not certain how to come up with one tho. :/
    # See more here:
    # https://scikit-learn.org/stable/tutorial/basic/tutorial.html
    clf = svm.SVC(kernel='poly', degree= 5, gamma = 'scale', C = 100)
    clf.fit(X_training,y_training)

    # Got the details of plotting from :
    # https://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Granularity of the partition visualization.
    step = 0.02
    # Boundries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Not sure what this 'ravel' thing is.
    # I geuss Z is the prediction space for each pixel?

    # print(Z.shape) - 61600
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # print(Z.shape) - 220,280 (which by the way is 61600.)
    plt.pcolormesh(xx,yy,Z,cmap = plt.cm.Paired)

    # Plot also the training points.
    plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Paired, edgecolors= 'k')
    plt.title('3-Class classification using Support Vector machine (but not custom Kernel!)')
    plt.axis('tight')
    plt.show()

    # Still has issues here.
    # # Extension I made that shows JUST Training
    # plt.pcolormesh(xx,yy,Z,cmap = plt.cm.Paired)
    # plt.scatter(X_training[:, 0], X_training[:, 1], c = y_training, cmap = plt.cm.Paired, edgecolors= 'k')
    # plt.title('3-Class classification using SVC. (Training Visual)')
    # #plt.axis('tight')
    # plt.show()
    #
    # # Extension I made that shows JUST Testing.
    # plt.pcolormesh(xx,yy,Z,cmap = plt.cm.Paired)
    # plt.scatter(X_testing[:, 0], X_testing[:, 1], c = y_testing, cmap = plt.cm.Paired, edgecolors= 'k')
    # plt.title('3-Class classification using SVC. (Testing Visual)')
    # #plt.axis('tight')
    # plt.show()

secondExploration()
