# -*- coding: utf-8 -*-
"""

Function for data scraping of NBA players advanced statistics

@author: DAndresSanchez

"""

def stats_season_adv(season_i, season_f=None):
    if season_f == None:
        season_f = season_i + 1

    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import pandas as pd

    # NBA season to analyse
    list_years = list(range(season_i, season_f))

    for year in list_years:
        # URL page to be scrapped
        url = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html".format(year)
        html = urlopen(url)
        soup = BeautifulSoup(html)

        # If it is the first loop, generate the headers  
        if year == list_years[0]:
            # Use getText() to extract the headers of the table into a list
            headers = [th.get_text() for th in soup.find_all('tr', limit=2)[0].find_all('th')]

            # Exclude the 'Rank' column
            headers = headers[1:]
            headers.insert(1, 'ID')

        # Avoid the header row 
        rows = soup.find_all('tr')[1:]

        # Generate an empty list for storing each player's stats    
        player_stats = []
        # Iterate over the rows of the table
        for i in range(len(rows)):
            tdarray = []  # list with 'td' tags, i.e. each player's stats
            for td in rows[i].find_all('td'):
                tdarray.append(td.get_text())
                for a in td.findAll('a'):  # get URL of player's stats per season
                    if 'teams' not in a['href']:
                        tdarray.append(a['href'].split('.')[0].split('/')[-1])
            # Append 'tdarray' to player_stats
            player_stats.append(tdarray)

        # Create dataframe from 'player_stats'
        year_stats = pd.DataFrame(player_stats, columns=headers)
        # Add column indicating season
        year_stats['Season'] = year
        # If first loop, create dataframe 'stats' from 'year_stats'
        if year == list_years[0]:
            stats = year_stats
        # If not first loop, append 'year_stats' to 'stats'
        else:
            stats = pd.concat([stats, year_stats], axis=0)

    # Data Cleaning
    # Create global ID
    stats['ID_Season'] = stats['ID'] + "_" + stats['Season'].astype(str)

    # Drop empty columns
    stats.drop(stats.columns[[19, 24]], axis=1, inplace=True)

    # Convert data type to numeric
    for col in list(stats.columns)[5:28]:
        stats[col] = pd.to_numeric(stats[col])

    # Remove rows with the 'Player' field empty or NaN
    stats.dropna(subset=['Player'], inplace=True)

    # Keep only Total rows in players who played in more than one team
    stats.drop_duplicates(subset="ID_Season", keep='first', inplace=True)
    stats.drop(columns=['Pos', 'Age', 'Tm'], inplace=True)

    for header in list(stats.columns):
        stats[header] = stats[header].fillna(0)

    # Order table by global ID and reset index
    stats.sort_values('ID_Season', inplace=True)
    stats.reset_index(drop=True, inplace=True)
    stats.set_index('ID_Season', inplace=True)

    return stats

