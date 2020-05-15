# -*- coding: utf-8 -*-
"""

Function for data scraping of NBA players advanced statistics

@author: DAndresSanchez


"""
     #%% Function definition
     
def stats_season(seasoni, seasonf=None):

    if seasonf == None:
        seasonf = seasoni + 1

     #%% Data Import
    
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import pandas as pd
    
    # NBA season to analyse
    list_years = list(range(seasoni,seasonf))

    for year in list_years:
        # URL page to be scrapped
        url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year)
        html = urlopen(url)
        soup = BeautifulSoup(html)
        
        # if it is the first loop, generate the headers  
        if year == list_years[0]:   
            # use getText() to extract the headers of the table into a list
            headers = [th.get_text() for th in soup.find_all('tr', limit=2)[0].find_all('th')]
            
            # exclude the 'Rank' column 
            headers = headers[1:]
            headers.insert(1,'ID')
        
        # avoid the header row 
        rows = soup.find_all('tr')[1:]
        
        # generate an empty list for storing each player's stats    
        player_stats = []   
        # iterate over the rows of the table
        for i in range(len(rows)):
            tdarray = [] # list with 'td' tags, i.e. each player's stats 
            for td in rows[i].find_all('td'):
                tdarray.append(td.get_text())
                for a in td.findAll('a'): # get URL of player's stats per season
                    if 'teams' not in a['href']:
                        tdarray.append(a['href'].split('.')[0].split('/')[-1])
            # append 'tdarray' to player_stats
            player_stats.append(tdarray)
        
        # create dataframe from 'player_stats'
        yearStats = pd.DataFrame(player_stats, columns = headers)
        # add column indicating season
        yearStats['Season'] = year
        # if first loop, create dataframe 'stats' from 'yearStats'
        if year == list_years[0]:
            stats = yearStats
        # if not first loop, append 'yearStats' to 'stats'
        else:
            stats = pd.concat([stats, yearStats], axis=0)
    
    
    
     #%% Data Cleaning
    
    pd.plotting.register_matplotlib_converters()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # create global ID
    stats['ID_Season'] = stats['ID']+ "_" + stats['Season'].astype(str)
    
    # convert data type to numeric
    for col in headers[5:]:
        stats[col]=pd.to_numeric(stats[col])
        
    # remove rows with the 'Player' field empty or NaN
    stats.dropna(subset=['Player'], inplace=True)
    
    # remove Total rows in players who played in more than one team
    stats.drop(stats[stats.Tm == 'TOT'].index, inplace=True)
    
    # when no atempts set value to 0
    for header in list(['FG%','3P%','2P%','eFG%','FT%']):
        stats[header] = stats[header].fillna(0)
    
    # order table by global ID and reset index
    stats.sort_values('ID_Season', inplace=True)
    stats.reset_index(drop=True, inplace=True)
     
    
    #%% Join data
    
    # define lambda function to perform weighted average
    wm = lambda x: np.average(x, weights=stats.loc[x.index, 'G'])
    
    aggdict = {
     'Player': 'first',
     'ID': 'first',
     'Pos': set,
     'Age': 'first',
     'Tm': set,
     'G': 'sum',
     'GS': 'sum',
     'MP': wm, 
     'FG': wm,
     'FGA': wm,
     'FG%': wm,
     '3P': wm,
     '3PA': wm,
     '3P%': wm,
     '2P': wm,
     '2PA': wm,
     '2P%': wm,
     'eFG%': wm,
     'FT': wm,
     'FTA': wm, 
     'FT%': wm,
     'ORB': wm,
     'DRB': wm,
     'TRB': wm,
     'AST': wm,
     'STL': wm,
     'BLK': wm,
     'TOV': wm,
     'PF': wm,
     'PTS': wm,
     'Season': 'first'
     }
    
    # group by global ID, merging players who played in >1 teams during 1 season
    statsbyplayer = stats.groupby('ID_Season', axis=0).agg(aggdict)
    
    return statsbyplayer

    #%% EDA
    
    print(statsbyplayer.head(10))
    print(statsbyplayer.shape)
    print(statsbyplayer.dtypes)
    print(statsbyplayer.describe())
    print(statsbyplayer.isnull().count())
    
    # graph showing the average of points recorded by James Harden in each season
    player='James Harden'
    df=statsbyplayer[statsbyplayer['Player']==player]
    df['Season'] = pd.to_datetime(df['Season'], format='%Y')
    
    plt.figure(figsize=(10,5))
    sns.scatterplot(x="Season", y="PTS", data=df)
    plt.xlabel('Season')
    plt.ylabel('Average Points per game')
    plt.title('Points per season of ' + player)
    plt.show()
    
    # graph showing how the percentage of games started affects the average of points per season
    plt.figure(figsize=(10,5))
    sns.scatterplot(statsbyplayer["GS"]/statsbyplayer["G"]*100, statsbyplayer["PTS"])
    plt.xlabel('%Games started')
    plt.ylabel('Average Points per game')
    plt.title('How the percentage of games started affects the average of points per season?')
    plt.show()
    
    
    #%% 
    
    
    df=statsbyplayer.loc[:,['MP','3P%','2P%','eFG%','FT%','PTS','AST','TRB']]
    df['%GS']=statsbyplayer.GS/statsbyplayer.G*100
    df['PTS']
    sns.pairplot(df);
    
    
    #%% Machine Learning
    
    
    
    #from sklearn.decomposition import PCA
    #model = PCA()
    df=statsbyplayer.iloc[:,5:]
    #model.fit(df)
    #print(model.components_)
    #transformed = model.transform(df)
    
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3)
    model.fit(df)
    labels = model.predict(df)
    
    import matplotlib.pyplot as plt
    xs = df.loc[:,'PTS']
    ys = df.loc[:,'AST']
    #plt.scatter(xs, ys, c=labels)
    
    #df2=statsbyplayer.loc[:,['MP','PTS','AST','TRB']]
    #from sklearn.cluster import KMeans
    #model = KMeans(n_clusters=3)
    #model.fit(df2)
    #labels = model.predict(df2)
    #df2['labels']=labels
    #sns.pairplot(df2, hue='labels');
    
    sns.PairGrid(df2, vars=['PTS', 'AST', 'TRB'],
                     hue='labels')
    
    
    
    
    
    
    
    
    
