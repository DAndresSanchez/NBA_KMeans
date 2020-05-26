# -*- coding: utf-8 -*-
"""

Clustering of NBA players based on their statistics.

Created on Mon May 11 14:46:21 2020

@author: DAndresSanchez

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.models import HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from nba_kmeans.stats_season import stats_season
from nba_kmeans.stats_season_adv import stats_season_adv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler

# Get the stats from season 2019
stats_bas = stats_season(2019)
stats_adv = stats_season_adv(2019)

# Join the basic and the advanced statistics and remove duplicate columns
stats = pd.concat([stats_bas, stats_adv], axis=1)
stats = stats.loc[:, ~stats.columns.duplicated()]

# Select only those players with more than 25 min played in more than 50 games
red_stats = stats[(stats['MP'] > 25) & (stats['G'] > 50)]

# Visualisation in Bokeh
# Define source for Bokeh graph
source = ColumnDataSource(data=dict(x=list(red_stats['USG%']),
                                    y=list(red_stats['PTS']),
                                    desc=list(red_stats['Player']),
                                    season=list(red_stats['Season'])))

# Define a hover as player and season
hover = HoverTool(tooltips=[
    ('Player', '@desc'),
    ('Season', '@season'),
])

# Define and show graph
plot = figure(plot_width=1000, plot_height=400, tools=[hover])
plot.circle('x', 'y', source=source, size=10, color="red", alpha=0.5)
plot.xaxis.axis_label = 'Usage %'
plot.yaxis.axis_label = 'Points'
output_file('USGvPoints.html')
show(plot)

# Determination of the optimal number of clusters
# Define dataframe to apply the KMeans algorithm
df = red_stats.loc[:, ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P', '3PA', '3P%', '2P',
                       '2PA', '2P%', 'eFG%', 'USG%']]

# Selection of optimal number of clusters:
inertia = {}
sil_coeff = {}
for k in range(2, 21):
    # Instantiate  KMeans and fit data
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=k)
    pipeline = make_pipeline(scaler, kmeans)
    pipeline.fit(df)
    label = kmeans.labels_
    # get inertia (Sum of distances of samples to their closest cluster center)
    inertia[k] = kmeans.inertia_
    # get silhouette score
    sil_coeff[k] = silhouette_score(df, label, metric='euclidean')

# Elbow Criterion Method: visualisation of inertia
plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(list(inertia.keys()), list(inertia.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.xticks(np.arange(2, 21, step=1))
plt.grid(linestyle='-', linewidth=0.5)

# Derivative of Inertia curve
plt.subplot(122)
plt.plot(list(inertia.keys()), np.gradient(list(inertia.values()), list(inertia.keys())))
plt.xlabel("Number of clusters")
plt.ylabel("Derivative of Inertia")
plt.xticks(np.arange(2, 21, step=1))
plt.grid(linestyle='-', linewidth=0.5)
plt.show()

# Silhouette Coefficient Method: visualisation silhouette scores
plt.figure(figsize=(7.5, 5))
plt.plot(list(sil_coeff.keys()), list(sil_coeff.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.xticks(np.arange(2, 21, step=1))
plt.grid(linestyle='-', linewidth=0.5)
plt.show()

# Comparison of preprocessing techniques for KMeans clustering
# KMeans clustering without preprocessing
# Define dataframe to apply the KMeans algorithm
df = red_stats.loc[:, ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P', '3PA', '3P%', '2P',
                       '2PA', '2P%', 'eFG%', 'USG%']]
n_clusters = 5
# Instantiate  KMeans and fit data
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df)
# Get clusters labels and assign them to dataframe
labels = kmeans.predict(df)
df['Labels'] = labels
# Visualisation
plt.figure(figsize=(16, 16))
plt.subplot(221)
plt.title('No preprocessing')
cmap = sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with StandardScaler
# Define dataframe to apply the KMeans algorithm
df = red_stats.loc[:, ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P', '3PA', '3P%', '2P',
                       '2PA', '2P%', 'eFG%', 'USG%']]
n_clusters = 5
# Instantiate  KMeans and fit data
scaler = StandardScaler()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(df)
# Get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels'] = labels
# Visualisation
plt.subplot(222)
plt.title('StandardScaler')
cmap = sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with Normalizer
# Define dataframe to apply the KMeans algorithm
df = red_stats.loc[:, ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P', '3PA', '3P%', '2P',
                       '2PA', '2P%', 'eFG%', 'USG%']]
# Instantiate  KMeans and fit data
norm = Normalizer()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(norm, kmeans)
pipeline.fit(df)
# Get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels'] = labels
# Visualisation
plt.subplot(223)
plt.title('Normalizer')
cmap = sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with MaxAbsScaler
# Define dataframe to apply the KMeans algorithm
df = red_stats.loc[:, ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P', '3PA', '3P%', '2P',
                       '2PA', '2P%', 'eFG%', 'USG%']]
# Instantiate  KMeans and fit data
maxabs = MaxAbsScaler()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(maxabs, kmeans)
pipeline.fit(df)
# Get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels'] = labels
# Visualisation
plt.subplot(224)
plt.title('MaxAbsScaler')
cmap = sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with StandardScaler and visualisation in Seaborn
# Define dataframe to apply the KMeans algorithm
df = red_stats.loc[:, ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P', '3PA', '3P%', '2P',
                       '2PA', '2P%', 'eFG%', 'USG%']]

# Instantiate  KMeans and fit data
n_clusters = 5
scaler = StandardScaler()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(df)

# Get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels'] = labels

cmap = sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# Plot main stats in a pair plot after clustering
sns.pairplot(df, vars=['PTS', 'USG%', 'TRB', 'AST'], hue='Labels', palette="muted")

# Visualisation in Bokeh after KMeans
# Define source for Bokeh graph
source = ColumnDataSource(data=dict(USG=list(red_stats['USG%']),
                                    PTS=list(red_stats['PTS']),
                                    AST=list(red_stats['AST']),
                                    desc=list(red_stats['Player']),

                                    season=list(red_stats['Season']),
                                    labels=list(map(str, list(labels)))
                                    ))

# Define a hover as player and season
hover = HoverTool(tooltips=[
    ('Player', '@desc'),
    ('Season', '@season'),
])

# Define the colors for mapping the labels from KMeans
mapper = CategoricalColorMapper(
    factors=[str(i + 1) for i in range(n_clusters)],
    palette=Category10[n_clusters])

# Define and show graph USG% vs PTS
plot = figure(plot_width=1000, plot_height=400, tools=[hover])
plot.circle('USG', 'PTS', source=source, size=10, alpha=0.75,
            color={'field': 'labels',
                   'transform': mapper})
plot.xaxis.axis_label = 'Usage %'
plot.yaxis.axis_label = 'Points'
output_file('USGvPTS.html')
show(plot)

# Define a hover as player and season
hover2 = HoverTool(tooltips=[
    ('Player', '@desc'),
    ('Season', '@season'),
])
# Define the colors for mapping the labels from KMeans
mapper2 = CategoricalColorMapper(
    factors=[str(i + 1) for i in range(n_clusters)],
    palette=Category10[n_clusters])
# Define source for Bokeh graph
source2 = ColumnDataSource(data=dict(USG=list(red_stats['USG%']),
                                     PTS=list(red_stats['PTS']),
                                     AST=list(red_stats['AST']),
                                     desc=list(red_stats['Player']),

                                     season=list(red_stats['Season']),
                                     labels=list(map(str, list(labels)))
                                     ))
# Define and show graph PTS vs AST
plot2 = figure(plot_width=1000, plot_height=400, tools=[hover2])
plot2.circle('AST', 'PTS', source=source2, size=10, alpha=0.75,
             color={'field': 'labels',
                    'transform': mapper2})
plot2.xaxis.axis_label = 'Assists'
plot2.yaxis.axis_label = 'Points'
output_file('ASTvPTS.html')
show(plot2)

# Define a hover as player and season
hover3 = HoverTool(tooltips=[
    ('Player', '@desc'),
    ('Season', '@season'),
])
# Define the colors for mapping the labels from KMeans
mapper3 = CategoricalColorMapper(
    factors=[str(i + 1) for i in range(n_clusters)],
    palette=Category10[n_clusters])
# Define source for Bokeh graph
source3 = ColumnDataSource(data=dict(USG=list(red_stats['USG%']),
                                     PTS=list(red_stats['PTS']),
                                     TRB=list(red_stats['TRB']),
                                     AST=list(red_stats['AST']),
                                     desc=list(red_stats['Player']),
                                     season=list(red_stats['Season']),
                                     labels=list(map(str, list(labels)))
                                     ))
# Define and show graph PTS vs AST
plot3 = figure(plot_width=1000, plot_height=400, tools=[hover3])
plot3.circle('TRB', 'PTS', source=source3, size=10, alpha=0.75,
             color={'field': 'labels',
                    'transform': mapper3})
plot3.xaxis.axis_label = 'Total Rebounds'
plot3.yaxis.axis_label = 'Points'
output_file('TRBvPTS.html')
show(plot3)

# Scree Plot for PCA
pca = PCA(n_components=10)
principal_components = pca.fit_transform(df)
principal_df = pd.DataFrame(data=principal_components
                            , columns=['principal component ' + str(e) for e in range(1, 11)])

# Scree plot to measure the weight of each principal component
scree = pd.DataFrame({'Variation': pca.explained_variance_ratio_,
                      'Principal Component': ['PC' + str(e) for e in range(1, 11)]})
sns.barplot(x='Principal Component', y='Variation',
            data=scree, color="c")
plt.title('Scree Plot')

# PC1 explains more than 75% of the variation
# PC1 and PC2 together account for almost 90% of the variation 
# Using PC1 and PC2 would be a good approximation


# PCA 2D visualisation
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df)
principal_df = pd.DataFrame(data=principal_components,
                            columns=['principal component 1', 'principal component 2'])
labels_no_index = list(df.Labels)
final_df = principal_df
final_df['labels'] = labels

# Define a hover as player and season
hover_pca = HoverTool(tooltips=[
    ('Player', '@desc'),
    ('Season', '@season'),
])
# Define the colors for mapping the labels from KMeans
mapper_pca = CategoricalColorMapper(
    factors=[str(i + 1) for i in range(n_clusters)],
    palette=Category10[n_clusters])
# Define source for Bokeh graph
source_pca = ColumnDataSource(data=dict(x=list(final_df['principal component 1']),
                                        y=list(final_df['principal component 2']),
                                        desc=list(red_stats['Player']),
                                        season=list(red_stats['Season']),
                                        labels=list(map(str, list(labels)))
                                        ))
# Define and show graph PC1 vs PC2
plot_pca = figure(plot_width=1000, plot_height=400, tools=[hover_pca])
plot_pca.circle('x', 'y', source=source_pca, size=10, alpha=0.75,
                color={'field': 'labels',
                       'transform': mapper_pca})
plot_pca.xaxis.axis_label = 'PC1'
plot_pca.yaxis.axis_label = 'PC2'
output_file('PCA.html')
show(plot_pca)
