### DESCRIPTION ###
# This python file is meant to be ran in a terminal to get a quick pred. of the premier
# league games in the following games
### ----------- ###


# Load packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scraping import scraper
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model
pd.set_option('display.max_columns', None)



# Scrape games of current season
season = 2023
Scraper = scraper(years = [season])
games = Scraper.fit(verbose = False)

# Keep games yet to be played between now and a week later
games = games[(games['Date'] >= datetime.today().strftime('%Y-%m-%d')) & (games['Date'] <= datetime.now() + timedelta(days = 7))]
games = games[(games['Result'].isna()) & (games['PTS%'].notna())]
games = games[['Date', 'Home', 'Away'] + Scraper.features].reset_index(drop = True)

if len(games) == 0 :
    print('No Premier League games schedule in the next week ...')

else :

    # Train the neural network
    model = load_model('NNMODEL.h5')

    # Predict the result of theupcoming games and append the results to the dataframe
    X_pred = games[Scraper.features]
    preds = model.predict(X_pred, verbose = 0)
    games['PRED.'] = list(pd.Series(np.argmax(preds, axis = 1)).replace(0, 'HOME').replace(1, 'DRAW').replace(2, 'AWAY'))
    games[['HOME_W', 'DRAW', 'AWAY_W']] = np.round(preds / preds.sum(axis = 1)[:,None], 2)
    games = games[['Date', 'Home', 'Away', 'PRED.', 'HOME_W', 'DRAW', 'AWAY_W']].rename(columns = {'Date':'DATE', 'Home':'HOME', 'Away':'AWAY'}).set_index('DATE')

    # Print the results
    print('\n\n\n')
    print(games)
    print('\n\n\n')