# -*- coding: utf-8 -*-
"""

Clustering of NBA players based on their statistics.

Created on Mon May 11 14:46:21 2020

@author: DAndresSanchez

"""

#%% Join tables

from stats_season import stats_season
from stats_season_adv import stats_season_adv
import pandas as pd

# get the stats from season 2019
stats_bas = stats_season(2019)
stats_adv = stats_season_adv(2019)

# join the basic and the advanced statistics and remove duplicate columns
stats = pd.concat([stats_bas, stats_adv], axis=1)
stats = stats.loc[:,~stats.columns.duplicated()]

# select only those players with more than 25 min played in more than 50 games
red_stats = stats[(stats['MP']>25) & (stats['G']>50)]

#%% Visualisation
    
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import LabelSet, Title, HoverTool

# define source for Bokeh graph
source = ColumnDataSource(data=dict(x=list(red_stats['USG%']),
                                    y=list(red_stats['PTS']),
                                    desc=list(red_stats['Player']),
                                    season=list(red_stats['Season'])))

# define a hover as player and season
hover = HoverTool(tooltips=[
        ('Player', '@desc'),
        ('Season', '@season'),
        ])

# define and show graph
plot = figure(plot_width=1000, plot_height=400, tools=[hover])
plot.circle('x', 'y', source=source, size=10, color="red", alpha=0.5)
plot.xaxis.axis_label = 'Usage %'
plot.yaxis.axis_label = 'Points'
output_file('USGvPoints.html')
show(plot)


#%% Machine Learning: KMeans clustering and visualisation in Seaborn
 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]

# initialise KMeans and fit data
model = KMeans(n_clusters=3)
model.fit(df)

# get clusters labels and assign them to dataframe
labels = model.predict(df)
df['labels']=labels

# plot main stats in a pair plot after clustering
sns.pairplot(df, vars=['PTS','USG%','TRB','AST'], hue='labels', palette="husl")

#%% Visualisation in Bokeh after KMeans
    
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import LabelSet, Title, HoverTool, CategoricalColorMapper
import pandas as pd

# define source for Bokeh graph
source = ColumnDataSource(data=dict(x=list(red_stats['USG%']),
                                    y=list(red_stats['PTS']),
                                    desc=list(red_stats['Player']),
                                    season=list(red_stats['Season']),
                                    labels=list(map(str, list(labels)))
                                    ))

# define a hover as player and season
hover = HoverTool(tooltips=[
        ('Player', '@desc'),
        ('Season', '@season'),
        ])

# define the colors for mapping the labels from KMeans
mapper = CategoricalColorMapper(
        factors=['0', '1', '2'],
        palette=['red', 'green', 'blue'])

# define and show graph
plot = figure(plot_width=1000, plot_height=400, tools=[hover])
plot.circle('x', 'y', source=source, size=10, alpha=0.5, 
            color={'field': 'labels',
                   'transform': mapper})
plot.xaxis.axis_label = 'Usage %'
plot.yaxis.axis_label = 'Points'
output_file('stats.html')
show(plot)


#%% Machine Learning: clustering
#
#from sklearn.cluster import SpectralClustering
#model = SpectralClustering(n_clusters=3)
#model.fit(df)
#labels = model.predict(df)
#print(model.inertia_)

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


