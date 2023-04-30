from search_engine import search
from scraping import scraper
from datetime import datetime, timedelta
from train import training, loading
import numpy as np
import pandas as pd

# PART 1
query = input('Enter competition :   ')
comp, flag = search(query)
st, code, name, country = comp['First Season'][0], comp['CODE'][0], comp['Competition'][0], comp['Country'][0]

# if flag :


# PART 2
# Scrape games of current season
season = 2023
Scraper = scraper(years = [season], code = code)
games = Scraper.fit(verbose = False)

# Keep games yet to be played between now and a week later
games = games[(games['Date'] >= datetime.today().strftime('%Y-%m-%d')) & (games['Date'] <= datetime.now() + timedelta(days = 7))]
games = games[(games['Result'].isna()) & (games['PTS%'].notna())]
games = games[['Date', 'Home', 'Away'] + Scraper.features].reset_index(drop = True)


if len(games) == 0 :
    print(f'No {name} ({country}) games scheduled in the next week ...')


else :

    try : # Try loading the model of the competition
        model = loading(code)

    except : # If the model doesn't yet exist, train one and save it

        model = training(code, st)

    # Predict the result of the upcoming games and append the results to the dataframe
    X_pred = games[Scraper.features]
    preds = model.predict(X_pred, verbose = 0)
    games['PRED.'] = list(pd.Series(np.argmax(preds, axis = 1)).replace(0, 'HOME').replace(1, 'DRAW').replace(2, 'AWAY'))
    games[['HOME_W', 'DRAW', 'AWAY_W']] = np.round(preds / preds.sum(axis = 1)[:, None], 2)
    games = games[['Date', 'Home', 'Away', 'PRED.', 'HOME_W', 'DRAW', 'AWAY_W']].rename(columns = {'Date':'DATE', 'Home':'HOME', 'Away':'AWAY'}).set_index('DATE')

    # Print the results
    print('\n\n\n')
    print(games)
    print('\n\n\n')