print('Importing Packages...')
import sys
sys.path.append('../../')
from utils.data import ModelData
from utils.util import mail
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd


json_name = './../../data/v1-b-all.json'

# Load CSV into Dataframe
print('Loading JSON...', end='\r')
df = pd.read_json(json_name)
print('Loading JSON... Done.')

# Split Data
mask = np.random.rand(len(df)) < 0.8
nozero = df.label != 0
Xtrain = np.array(df[mask & nozero].X.tolist())
ytrain = np.array(df[mask & nozero].label.tolist()) - 1
Xtest = np.array(df[~mask & nozero].X.tolist())
ytest = np.array(df[~mask & nozero].label.tolist()) - 1

print('Shape of input:', Xtrain.shape)

print('Building MLP Model...')
activation = 'softmax'
model = models.Sequential([
    layers.Flatten(input_shape=Xtrain.shape[1:]),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(Xtrain.shape[-1], activation=activation),
    layers.Dense(7, activation='softplus')
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

csv_name = f'./../../outputs/v1-b/activation_study/varied-epoch-{activation}-sp_out.csv'
df.to_csv(csv_name, index=False)

message = 'Subject: {}\n\n{}'.format('ML Model Update',
                                     f'{csv_name.split("/")[-1]} finished computing...'
                                     f'\n\n* * * * Results * * * *\n'
                                     f'{df}'
                                     f'\n\nThanks! =(^_^)=')

mail(['bdaghir@gatech.edu'], message)