print('Importing Packages...')
import sys
sys.path.append('../')
from utils.data import ModelData
from utils.util import mail
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd


json_name = './data/v1-b-all.json'


# # Script for creating the data frame for G00
# print('Creating Data Objects...')
# path = '/Users/benjamindaghir1/Dropbox (GaTech)/CS4641/data'
# data = ModelData(path)
#
# print('Loading Group 00 Data...')
# g00 = data.G00
# g00.load_group_data()
#
# print('Extracting Field ID Data...')
# # Preparing Dataframe to CSV
# df00 = g00.extract_group_fids()
# df00 = df00.sort_values(['label']).reset_index()
# print('Writing JSON...', end='\r')
# df00.to_json(json_name)
# print('Writing JSON... Done.')
#
#
# # Script for creating the data frame for all data
# print('Creating Data Objects...')
# path = '/Users/benjamindaghir1/Dropbox (GaTech)/CS4641/data'
# data = ModelData(path)
#
# print('Loading All Data...')
# data.load_all_data()
#
# print('Extracting Field ID Data...')
# # Preparing Dataframe to CSV
# df = data.extract_fids()
# df = df.sort_values(['label']).reset_index()
# print('Writing JSON...', end='\r')
# df.to_json(json_name)
# print('Writing JSON... Done.')


# Load CSV into Dataframe
print('Loading JSON...', end='\r')
df = pd.read_json(json_name)
print('Loading JSON... Done.')

# TODO:// Manipulate Data


# Split Data
mask = np.random.rand(len(df)) < 0.8
nozero = df.label != 0
Xtrain = np.array(df[mask & nozero].X.tolist())
ytrain = np.array(df[mask & nozero].label.tolist()) - 1
Xtest = np.array(df[~mask & nozero].X.tolist())
ytest = np.array(df[~mask & nozero].label.tolist()) - 1

print('Shape of input:', Xtrain.shape)

print('Building MLP Model...')
model = models.Sequential([
    layers.Flatten(input_shape=Xtrain.shape[1:]),
    layers.Dense(Xtrain.shape[-1], activation='relu'),
    layers.Dense(Xtrain.shape[-1], activation='relu'),
    layers.Dense(Xtrain.shape[-1], activation='relu'),
    layers.Dense(Xtrain.shape[-1], activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

losses = []
accuracies = []
epochs = [1, 5, 10, 50, 100, 500, 1000]
for e in epochs:
    model.fit(Xtrain, ytrain, epochs=e)
    out = model.evaluate(Xtest, ytest)
    losses.append(out[0])
    accuracies.append(out[1])

print('\n* * * Data Summarized * * *')
for i in range(len(epochs)):
    print('Epochs:', epochs[i], 'Loss:', losses[i], 'Accuracy:', accuracies[i])

df = pd.DataFrame({
    'epochs': epochs,
    'loss': losses,
    'accuracy': accuracies
})

df.to_csv('./outputs/v1-b/varied-epoch-4HL.csv', index=False)

mail(['bdaghir@gatech.edu'])