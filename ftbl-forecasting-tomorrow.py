# This python file is meant to be ran in a terminal to create quick predictions of football game happening tomorrow
# in the football world
# When prompted in a terminal this file print the matches and their respective predictions


# Load packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from email_results import send
from scraping import scraper
from datetime import datetime, timedelta
from train import training, loading
import numpy as np
import pandas as pd

# Load the competitions
competitions = pd.read_csv('competitions.csv')

# Initiate results dictionary
results = {}
# Loop through all covered competitions
for code, name, country in zip(competitions['CODE'], competitions['Competition'], competitions['Country']) :

    print(f"\nScraping tomorrow's {name} ({country}) games ...")
    Scraper = scraper(code = code)
    games = Scraper.fit_pred()
    # Keep games yet to be played between now and a week later
    today = datetime(datetime.today().year, datetime.today().month, datetime.today().day)
    now, then = today, today + timedelta(days = 1)
    games = games[(games['Date'] >= now) & (games['Date'] < then)]
    games = games[(games['Result'].isna()) & (games['PTS%'].notna())]
    games = games[['Date', 'Home', 'Away'] + Scraper.features].reset_index(drop = True)

    # Stop if there are no games scheduled in the upcoming week
    if len(games) == 0 :
        print(f'No {name} ({country}) games scheduled tomorrow ...')


    # PART 3 :   Predict the results of the upcoming games
    else :

        try :     # Try loading the model of the competition
            model = loading(code)

        except :  # If the model doesn't yet exist, train one and save it
            model = training(code)

        # Predict the result of the upcoming games and append the results to the dataframe
        X_pred = np.array(games[Scraper.features])
        preds = model.predict(X_pred, verbose = 0)
        games['PRED.'] = list(pd.Series(np.argmax(preds, axis = 1)).replace(0, 'HOME').replace(1, 'DRAW').replace(2, 'AWAY'))
        games[['HOME_W', 'DRAW', 'AWAY_W']] = np.round(preds / preds.sum(axis = 1)[:, None], 2)
        games = games[['Date', 'Home', 'Away', 'PRED.', 'HOME_W', 'DRAW', 'AWAY_W']].rename(columns = {'Date':'DATE', 'Home':'HOME', 'Away':'AWAY'})

        # Send by email
        subject = f'{name.upper()} ({country.upper()}) PREDICTIONS ({games["DATE"].min().strftime("%d %b")} - {games["DATE"].max().strftime("%d %b")})\n'
        # send(games, subject)
        
        # Save the results
        results[f'{name.upper()} ({country})'] = games.set_index('DATE')

print('\n\n\n\n\n\n\n')
print("  TOMORROW'S FOOTBALL PREDICTIONS  :\n")
for prompt, df in results.items():

    print(f'\n\t{prompt}')
    print(f'{df}\n')

print('\n\n\n')