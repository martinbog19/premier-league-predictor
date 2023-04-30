import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scraping import scraper
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

def training(code, st, acyear) :

    season = 2023

    try :     # Load training data
        train = pd.read_csv(f'training_data/TRAINING_DATA_COMP{code}.csv')
        print('Loading training data ...')
        Scraper = scraper(years = [season], code = code, acyear = acyear)
        _ = Scraper.fit()

    except :  # If it doesn't exist, scrape it and save it
        print('No training data found ... ')
        years = np.arange(np.max([1980, st]), season)
        Scraper = scraper(years = years, code = code, acyear = acyear)
        train = Scraper.fit()
        train.to_csv(f'training_data/TRAINING_DATA_COMP{code}.csv', index = None)
        print('Training data built ... ')

    print('No model found ... building neural network model ... ')
    # Clean training data -- at least 5 games played in the season + no games from COVID-19 1st lockdown
    train = train[(train['Game_home'] > 4) & (train['Game_away'] > 4) & (train['Rest'].abs() <= 40)]
    # Shuffle data
    train = shuffle(train)
    # Set X and y matrices
    X_train = np.array(train[Scraper.features])
    y_train = np.array(train[['Win', 'Draw', 'Loss']])

    # Create neural network model
    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = len(Scraper.features)))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3,  activation = 'softmax'))

    # Compile model
    model.compile(loss      =  'categorical_crossentropy',
                  optimizer =  'adam',
                  metrics   = ['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs = 50, batch_size = 32, validation_split = 0.1, verbose = 0)

    # Save model
    model.save(f'models/MODEL_COMP{code}.h5')

    return model


def loading(code) :

    model = load_model(f'models/MODEL_COMP{code}.h5')
    print('Loading neural network model ...')
    return model