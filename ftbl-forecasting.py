import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from search_engine import search
from scraping import scraper
from datetime import datetime, timedelta
from train import training, loading
import numpy as np
import pandas as pd

print('\n')
# PART 1 :   Ask the user for a competition
query = input('Enter competition :   \n\t')
comp, flag = search(query)
print('\n')


if flag :

    st, code, name, country, acyear = comp['First Season'][0], comp['CODE'][0], comp['Competition'][0], comp['Country'][0], comp['acyear'][0]
    # PART 2 :   Scrape games to be played in the next week
    season = 2023
    print('Scraping next few games ...')
    Scraper = scraper(years = [season], code = code)
    games = Scraper.fit(verbose = False)
    # Keep games yet to be played between now and a week later
    now, then = datetime.today().strftime('%Y-%m-%d'), datetime.now() + timedelta(days = 7)
    games = games[(games['Date'] >= now) & (games['Date'] <= then)]
    games = games[(games['Result'].isna()) & (games['PTS%'].notna())]
    games = games[['Date', 'Home', 'Away'] + Scraper.features].reset_index(drop = True)

    # Stop if there are no games scheduled in the upcoming week
    if len(games) == 0 :
        print(f'No {name} ({country}) games scheduled in the next week ...')


    # PART 3 :   Predict the results of the upcoming games
    else :

        try :     # Try loading the model of the competition
            model = loading(code)

        except :  # If the model doesn't yet exist, train one and save it
            model = training(code, st, acyear)

        # Predict the result of the upcoming games and append the results to the dataframe
        X_pred = np.array(games[Scraper.features])
        preds = model.predict(X_pred, verbose = 0)
        games['PRED.'] = list(pd.Series(np.argmax(preds, axis = 1)).replace(0, 'HOME').replace(1, 'DRAW').replace(2, 'AWAY'))
        games[['HOME_W', 'DRAW', 'AWAY_W']] = np.round(preds / preds.sum(axis = 1)[:, None], 2)
        games = games[['Date', 'Home', 'Away', 'PRED.', 'HOME_W', 'DRAW', 'AWAY_W']].rename(columns = {'Date':'DATE', 'Home':'HOME', 'Away':'AWAY'}).set_index('DATE')

        # Print the results
        print('\n')
        print(f'{name.upper()} ({country.upper()}) PREDICTIONS ({games.index.min().strftime("%d %b")} - {games.index.max().strftime("%d %b")})\n')
        print(games)
        print('\n\n')