# -*- coding: utf-8 -*-
"""

Function for data scraping of NBA players advanced statistics

@author: DAndresSanchez


"""

    #%% Function definition
     
def stats_season_adv(seasoni, seasonf=None):

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
        url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)
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
    
    # drop empty columns
    stats.drop(stats.columns[[19, 24]], axis=1, inplace=True)
    
    # convert data type to numeric
    for col in list(stats.columns)[5:28]:
        stats[col]=pd.to_numeric(stats[col])
        
    # remove rows with the 'Player' field empty or NaN
    stats.dropna(subset=['Player'], inplace=True)
    
    # keep only Total rows in players who played in more than one team
    stats.drop_duplicates(subset ="ID_Season", keep = 'first', inplace = True) 
    stats.drop(columns=['Pos','Age','Tm'], inplace=True)
    
    for header in list(stats.columns):
        stats[header] = stats[header].fillna(0)
    
    # order table by global ID and reset index
    stats.sort_values('ID_Season', inplace=True)
    stats.reset_index(drop=True, inplace=True)
    stats.set_index('ID_Season', inplace=True)
    
    return stats

    #%% EDA
    
    print(stats.head(10))
    print(stats.shape)
    print(stats.dtypes)
    print(stats.describe())
    print(stats.isnull().sum())
    print(stats.isna().sum())
    
    ## graph showing the average of points recorded by James Harden in each season
    #player='James Harden'
    #df=statsbyplayer[statsbyplayer['Player']==player]
    #df['Season'] = pd.to_datetime(df['Season'], format='%Y')
    #
    #plt.figure(figsize=(10,5))
    #sns.scatterplot(x="Season", y="PTS", data=df)
    #plt.xlabel('Season')
    #plt.ylabel('Average Points per game')
    #plt.title('Points per season of ' + player)
    #plt.show()
    #
    ## graph showing how the percentage of games started affects the average of points per season
    #plt.figure(figsize=(10,5))
    #sns.scatterplot(statsbyplayer["GS"]/statsbyplayer["G"]*100, statsbyplayer["PTS"])
    #plt.xlabel('%Games started')
    #plt.ylabel('Average Points per game')
    #plt.title('How the percentage of games started affects the average of points per season?')
    #plt.show()
    
    
    
    
    #%% Machine Learning: clustering
    
    df=stats.iloc[:,2:]
    
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3)
    model.fit(df)
    labels = model.predict(df)
    
    
    df['labels']=labels
    
    sns.pairplot(df, vars=['PER','USG%','TS%','BPM'], hue='labels')
    
    
    #%% Machine Learning: PCA visualisation
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    labelsNoIndex=list(df.labels)
    finalDf= principalDf
    finalDf['labels'] = labels
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    label=[0, 1, 2]
    colors = ['r', 'g', 'b']
    for label, color in zip(label,colors):
        indicesToKeep = finalDf['labels'] == label
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(label)
    ax.grid()
    
    
