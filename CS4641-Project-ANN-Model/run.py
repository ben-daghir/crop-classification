import sys
from utils.data import ModelData
from utils.util import mail
from studies.CropModel import ModelWrapper, Dataset
import pandas as pd
import time

from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops

import numpy as np
import matplotlib.pyplot as pl

from sklearn import decomposition
import csv

DATA_DATES = [ \
        "06-06-2019", \
        "07-01-2019", \
        "07-06-2019", \
        "07-11-2019", \
        "07-21-2019", \
        "08-05-2019", \
        "08-15-2019", \
        "08-25-2019", \
        "09-09-2019", \
        "09-19-2019", \
        "09-24-2019", \
        "10-04-2019", \
        "11-03-2019"]

def apply_pca(X, num_pca_features=4):
    print('Feature reducing dataset...')
    pca = decomposition.PCA(n_components=num_pca_features)
    num_pixels, num_bands, num_dates = X.shape
    X_pca = np.zeros((num_pixels, num_pca_features, num_dates))
    for date_idx in range(num_dates):
        pca.fit(X[:,:,date_idx])
        X_pca[:,:,date_idx] = pca.transform(X[:,:,date_idx])
        print('Dataset of {} retained {}% of variance'.format( \
                DATA_DATES[date_idx], \
                pca.explained_variance_ratio_.cumsum()[-1] * 100))

    return X_pca

def main():
    json_path = './data/v1-b-all.json'
    df = pd.read_json(json_path)

    mask = np.random.rand(len(df)) < 0.8
    nozero = df.label != 0
    X = np.array(df[nozero].X.tolist())
    X = np.delete(X, 12, 1) # delete the CLD data and normalize
    Y = np.array(df[nozero].label.tolist())

    single_dense = ModelWrapper(Dataset(X, Y))
    single_dense.set('model', keras.Sequential([ \
            keras.layers.Flatten(input_shape=single_dense.dataset.xtrain.shape[1:]), \
            keras.layers.Dense(128, activation=tf.nn.relu), \
            keras.layers.Dense(64, activation=tf.nn.relu), \
            keras.layers.Dense(32, activation=tf.nn.relu), \
            keras.layers.Dense(14, activation=tf.nn.relu), \
            keras.layers.Dense(7, activation=tf.nn.softmax) \
            ]))
    single_dense.get('model').compile( \
        optimizer='adam', \
        loss='sparse_categorical_crossentropy', \
        metrics=['accuracy'])
    history = single_dense.train(epochs=500, verbose=0)
    test_loss, test_acc = single_dense.evaluate(verbose=0)

    print(np.array(history.history["accuracy"]))
    log = np.stack([np.array(history.history["accuracy"]), np.array(history.history["loss"])])
    #test_epochs=[10, 50, 100, 250, 500]
    #batch_sizes=np.rint(np.array([1, 0.25, 0.05, 0.0025]) * single_dense.dataset.xtrain.shape[0])
    #batch_sizes=np.append(batch_sizes, np.array([64, 32, 8, 4, 2]))
    #default_e=250

    # Using the default batch size of 32, test accuracy as function of epochs
    #acc_v_epochs = np.zeros((2, len(test_epochs)))
    #for e in range(len(test_epochs)):
    #    ops.reset_default_graph()
    #    single_dense.get('model').compile( \
    #        optimizer='adam', \
    #        loss='sparse_categorical_crossentropy', \
    #        metrics=['accuracy'])
    #    single_dense.train(epochs=test_epochs[e], verbose=0)
    #    test_loss, test_acc = single_dense.evaluate(verbose=0)
    #    acc_v_epochs[0, e] = test_loss
    #    acc_v_epochs[1, e] = test_acc
    #    print("loss: {} accuracy: {}".format(test_loss, test_acc))

    # Using epochs = 250, test accuracy as function of the batch size
    #acc_v_batch_size = np.zeros((3, len(batch_sizes)))
    #for b in range(len(batch_sizes)):
    #    print("Testing batchsize {}".format(int(batch_sizes[b])))
    #    start_time = time.time()
    #    ops.reset_default_graph()
    #    single_dense.get('model').compile( \
    #        optimizer='adam', \
    #        loss='sparse_categorical_crossentropy', \
    #        metrics=['accuracy'])
    #    single_dense.train(epochs=default_e, verbose=0, batch_size=int(batch_sizes[b]))
    #    test_loss, test_acc = single_dense.evaluate(verbose=0)
    #    acc_v_batch_size[0, b] = test_loss
    #    acc_v_batch_size[1, b] = test_acc
    #    acc_v_batch_size[2, b] = time.time() - start_time
    #    print("loss: {} accuracy: {}".format(test_loss, test_acc))

    #acc_v_pca_dims = np.zeros((3, 8))
    ### Using the default batch size, test the result with the number of dimensions
    #for n in range(0, 8):
    #    print("Testing pca with dims {}".format(n + 1))
    #    start_time = time.time()
    #    model = ModelWrapper(Dataset(X, Y, preprocess=[partial(apply_pca, num_pca_features=n + 1)]))
    #    model.set('model', keras.Sequential([ \
    #            keras.layers.Flatten(input_shape=model.dataset.xtrain.shape[1:]), \
    #            keras.layers.Dense(128, activation=tf.nn.relu), \
    #            keras.layers.Dense(64, activation=tf.nn.relu), \
    #            keras.layers.Dense(7, activation=tf.nn.softmax) \
    #            ]))
    #    ops.reset_default_graph()
    #    model.get('model').compile( \
    #        optimizer='adam', \
    #        loss='sparse_categorical_crossentropy', \
    #        metrics=['accuracy'])
    #    model.train(epochs=default_e, verbose=0)
    #    test_loss, test_acc = model.evaluate(verbose=0)
    #    acc_v_pca_dims[0, n] = test_loss
    #    acc_v_pca_dims[1, n] = test_acc
    #    acc_v_pca_dims[2, n] = time.time() - start_time
    #    print("loss: {} accuracy: {}".format(test_loss, test_acc))

    #np.savetxt("acc_v_pca_dims.csv", acc_v_pca_dims, delimiter=",")
    #np.savetxt("acc_v_epochs.csv", acc_v_epochs, delimiter=",")
    #np.savetxt("acc_v_batch_size.csv", acc_v_batch_size, delimiter=",")
    np.savetxt("training_stats.csv", log, delimiter=",")

if __name__ == "__main__":
    main()
