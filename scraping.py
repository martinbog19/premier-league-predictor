import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from time import sleep
import sys


class scraper:


    def __init__(self, code) :
        self.code   = code


    def scrape_year(self, soup, windows = [1, 2, 5, 10]) : # Transform yearly soup to dataframe of games

        while soup.find('tr', class_ = 'thead') is not None: # Decompose all headers
                soup.find('tr', class_ = 'thead').decompose()

        data = pd.read_html(str(soup.find('table')))[0] # Read table in a dataframe
        data = data[data['Home'].notna()] # Remove headers
        if 'Notes' in data.columns : # Remove 'special' games
            data = data[data['Notes'].isna()]
        data = data[['Date', 'Home', 'Away', 'Score']] # Only keep necessary columns
        data['Date'] = pd.to_datetime(data['Date']) # Turn dates into datetime format
        data['G_home'] = data['Score'].apply(lambda x: int(str(x).split('–')[0])  if (np.all(pd.notnull(x))) else x) # Set home goals
        data['G_away'] = data['Score'].apply(lambda x: int(str(x).split('–')[-1]) if (np.all(pd.notnull(x))) else x) # Set away goals
            
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
        # Calculate the differentials for each feature: HOME - AWAY
        for f in self.features:
            games[f] = games[f'{f}_home'] - games[f'{f}_away']
        # Keep the relevant columns -- re-order neatly
        games = games.rename(columns = {'W_home': 'Result', 'GF_home': 'G_home', 'GA_home': 'G_away'})
        games = games[['Date', 'Game_home', 'Game_away','Home', 'Away', 'Result', 'G_home', 'G_away'] + self.features]
        return games



    def fit_pred(self) :

        # Scrape current season's schedule
        url = f'https://fbref.com/en/comps/{self.code}/schedule/'
        page = requests.get(url)
        if page.status_code == 429 :
            sys.exit("I'm being rate limited ...")
        if page.status_code == 500 :
            sys.exit("The page is not accessible ...")
        soup = BeautifulSoup(page.content, 'lxml') # Create a soup object from the webpage
        if soup.find('table') is not None :
            DATA = self.scrape_year(soup)

        # Create output matrix
        DATA = pd.concat([DATA, pd.get_dummies(DATA['Result'].replace(1,'Win').replace(0,'Draw').replace(-1,'Loss'))], axis = 1)
        DATA = DATA.sort_values('Date').reset_index(drop = True)

        return DATA

    def fit_train(self, verbose = True) :

        # Scrape the table with all seasons from the competition
        url = f'https://fbref.com/en/comps/{self.code}/history/'
        page = requests.get(url) # Get the webpage
        if page.status_code == 429 :
            sys.exit("I'm being rate limited ...")
        if page.status_code == 500 :
            sys.exit("The page is not accessible ...")
        soup = BeautifulSoup(page.content, 'lxml') # Create soup object from the webpage
        table = soup.find('table') # Find the table
        seasons = pd.read_html(str(table))[0] # Create a dataframe from the data
        seasons['href'] = [x['href'] for x in table.find_all('a', href = True) if 'comps' in x['href']][0::2] # Get the url of each season
        seasons['Year'] = seasons['Season'].apply(lambda x: x.split('-')[-1]).astype(int) # Define the year of the season
        seasons = seasons[seasons['Year'] > 1980] # Only keep seasons from 1980
        seasons = seasons.sort_values('Year').reset_index(drop = True)

        yearly_data = []
        for href, season in zip(seasons['href'], seasons['Season']) : # Loop through the seasons

            # Create the url for the looped season
            url = 'https://fbref.com' + '/'.join(href.split('/')[:-1]) + '/schedule/'
            page = requests.get(url) # Get the webpage
            soup = BeautifulSoup(page.content, 'lxml') # Create soup object from the webpage
            header = page.text.split('</h1>')[0].split('<h1>')[-1] # Find the header of the page
            title = header.split(' ')[-1].split('\n')[0] # Find the type of data the page contains

            if verbose :
                print(f'Building training data ... {season} ...        ', end = '\r')

            # If the page contains a table and its title is fictures, scrape this table
            if soup.find('table') is not None and title == 'Fixtures' :
                
                games = self.scrape_year(soup)
                yearly_data.append(games)

            sleep(3) # Pause for 3 seconds to avoid getting rate limited

        # Take away the last element of the yearly data, it is the current season
        yearly_data = yearly_data[:-1]
        # Finally, concatenate all seasons together
        DATA = pd.concat(yearly_data)
        # Create output matrix
        DATA = pd.concat([DATA, pd.get_dummies(DATA['Result'].replace(1,'Win').replace(0,'Draw').replace(-1,'Loss'))], axis = 1)
        DATA = DATA.sort_values('Date').reset_index(drop = True)

        return DATA
