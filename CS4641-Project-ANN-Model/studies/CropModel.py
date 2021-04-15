import numpy as np
import tensorflow as tf

class ModelWrapper:
    def __init__(self, dataset, description=None, model=None):
        self.graph = tf.Graph()
        if model:
            self.model = model
        self.dataset = dataset
        self.description = description

    def train(self, **kwargs):
        return self.model.fit(self.dataset.xtrain, self.dataset.ytrain, **kwargs)

    def evaluate(self, **kwargs):
        return self.model.evaluate(self.dataset.xtest, self.dataset.ytest, **kwargs)

    def get(self, attr):
        return getattr(self, attr)

    def set(self, attr_name, attr):
        setattr(self, attr_name, attr)

class Dataset:
    def __init__(self, X, Y, batch_size=0.8, preprocess=None):
        self.X = X
        self.Y = Y
        if preprocess:
            for p in preprocess:
                print(self.X.shape)
                self.X = p(self.X)
        mask = np.random.rand(X.shape[0]) < 0.8
        self.xtrain, self.xtest = X[mask], X[~mask]
        self.ytrain, self.ytest = Y[mask] - 1, Y[~mask] - 1
