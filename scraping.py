import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from IPython.display import clear_output


class scraper:


    def __init__(self, years) :
        self.years = years


    def fit(self, windows = [1, 2, 5, 10]) :

        yearly_data = []
        for i, year in enumerate(self.years) :

            print(f'... {year-1}-{year} ...  ({i+1}/{len(self.years)}) ...')

            # Scrape looped year's schedule
            url = f'https://fbref.com/en/comps/9/{year-1}-{year}/schedule/'
            soup = BeautifulSoup(requests.get(url).content, 'lxml') # Create a soup object from the webpage
            while soup.find('tr', class_ = 'thead') is not None: # Decompose all headers
                soup.find('tr', class_ = 'thead').decompose()
            data = pd.read_html(str(soup.find('table')))[0] # Read table in a dataframe
            data = data[data['Score'].notna()] # Remove headers
            data = data[['Date', 'Home', 'Away', 'Score']] # Only keep necessary columns
            data['Date'] = pd.to_datetime(data['Date']) # Turn dates into datetime format
            data['G_home'] = data['Score'].apply(lambda x: str(x).split('–')[0]).astype(int)  # Set home goals
            data['G_away'] = data['Score'].apply(lambda x: str(x).split('–')[-1]).astype(int) # Set away goals

            # Create list of teams competing this season
            teams = sorted(list(set(data['Home'])))
            homes, aways = [], [] # Initiate empty lists of home & away games
            # For each team keep their games and compute the rolling -- with varying windows -- statistics
            # Separate into home and away games
            for tm in teams:

                df = data.copy()[(data['Home'] == tm) | (data['Away'] == tm)].sort_values('Date').reset_index(drop = True) # Team results
                df['H'] = df['Home'] == tm # Differentiate home & away games
                df['A'] = df['H']    == False
                df['Team'] = len(df) * [tm] # Specify home team
                df['Opp'] = df['Home'] * df['A'] + df['Away'] * df['H'] # Specify name of opponent
                df['GF'] = df['G_home'] * df['H'] + df['G_away'] * df['A'] # Goals for
                df['GA'] = df['G_home'] * df['A'] + df['G_away'] * df['H'] # Goals against
                df['GD'] = df['GF'] - df['GA'] # Goal difference
                df['W'] = np.sign(df['GD']) # Win (1), Draw (0) or Loss (-1)
                df = df[['Date', 'Team', 'Opp', 'H', 'W', 'GF', 'GA', 'GD']] # Keep relevant columns
                df['PTS%'] = [np.nan] + list(np.cumsum(df['W'].apply(lambda x: {-1:0,0:1,1:3}.get(x))) / (3*(df.index + 1)))[:-1] # Calculate the % of max. PTS
                self.features = ['PTS%'] # Add it as a features
                for f in ['W', 'GF', 'GA', 'GD']: # Loop for all features
                    for w in windows: # and windows
                        # Compute the lagged feature
                        df[f'{f}_{w}'] = [np.nan] + list(df[f].rolling(w, min_periods = 1).mean())[:-1]
                        self.features.append(f'{f}_{w}') # Add lagged features to features list
                    # Compute the rolling mean for each feature
                    df[f'{f}_inf'] = [np.nan] + list(df[f].rolling(1000, min_periods = 1).mean())[:-1]
                    self.features.append(f'{f}_inf') # Add it again to the list of features
                df['Rest'] = [np.nan] + list(np.diff(df['Date']).astype(float) / (10**9) / (60 * 60 * 24)) # Compute the number of days of rest
                self.features.append('Rest') # Add to list of features
                df['Game'] = df.index + 1 # Specify the number of games played

                # Cluster home games and rename columns accordingly
                home = df.groupby('H').get_group(True)
                renameDict = {'Team': 'Home', 'Opp': 'Away'}
                for f in ['W', 'GF', 'GA', 'GD', 'Game'] + self.features:
                    renameDict[f] = f'{f}_home'
                home = home.rename(columns = renameDict)
                # Cluster away games and rename columns accordingly
                away = df.groupby('H').get_group(False)
                renameDict = {'Team': 'Away', 'Opp': 'Home'}
                for f in ['W', 'GF', 'GA', 'GD', 'Game'] + self.features:
                    renameDict[f] = f'{f}_away'
                away = away.rename(columns = renameDict)
                away = away.drop(columns = ['H', 'W_away', 'GF_away', 'GA_away', 'GD_away']) # Drop redundant columns
                # Append home & away games to appropriate lists
                homes.append(home)
                aways.append(away)

            # Each of the 380 games has a copy in home & away -- merge them to get the complete dataframe of all games
            home = pd.concat(homes)
            away = pd.concat(aways)
            games = home.merge(away, on = ['Date', 'Home', 'Away']).sort_values('Date').reset_index(drop = True).drop(columns = ['H'])
            games['Season'] = len(games) * [year] # Keep track of seasons
            # Calculate the differentials for each feature: HOME - AWAY
            for f in self.features:
                games[f] = games[f'{f}_home'] - games[f'{f}_away']
            # Keep the relevant columns -- re-order neatly
            games = games.rename(columns = {'W_home': 'Result', 'GF_home': 'G_home', 'GA_home': 'G_away'})
            games = games[['Date', 'Season', 'Game_home', 'Game_away','Home', 'Away', 'Result', 'G_home', 'G_away'] + self.features]
            # Append games to the list of yearly games
            yearly_data.append(games)
            clear_output(wait = True)

        # Finally, concatenate all seasons together
        DATA = pd.concat(yearly_data)
        DATA = DATA.sort_values('Date').reset_index(drop = True)

        return DATA