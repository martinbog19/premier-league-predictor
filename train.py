from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np

def train_model() :

    # Load training data
    train = pd.read_csv('data_1980_2022.csv')
    # Clean training data -- at least 5 games played in the season + no games from COVID-19 1st lockdown
    train = train[(train['Game_home'] > 4) & (train['Game_away'] > 4) & (train['Rest'].abs() < 30)]
    # Set X and y matrices
    X_train = np.array(train[train.columns[9:-3]])
    y_train = np.array(train[['Win', 'Draw', 'Loss']])

    # Create neural network model
    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = len(train.columns[9:-3])))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3,  activation = 'softmax'))

    # Compile model
    model.compile(loss      =  'categorical_crossentropy',
                  optimizer =  'adam',
                  metrics   = ['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs = 50, batch_size = 32, validation_split = 0.1, verbose = 0)

    model.save('NNMODEL.h5')

train_model()