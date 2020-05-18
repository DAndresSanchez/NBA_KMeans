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

#%% Machine Learning: optimal number of clusters

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]
     
# selection of optimal number of clusters:
inertia={}
sil_coeff={}
for k in range(2,21):
    # initialise KMeans and fit data
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
plt.figure(figsize=(16,5))
plt.subplot(121)
plt.plot(list(inertia.keys()), list(inertia.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.xticks(np.arange(2, 21, step=1))
plt.grid(linestyle='-', linewidth=0.5)
# derivative of Inertia curve
plt.subplot(122)
plt.plot(list(inertia.keys()),np.gradient(list(inertia.values()),list(inertia.keys())))
plt.xlabel("Number of clusters")
plt.ylabel("Derivative of Inertia")
plt.xticks(np.arange(2, 21, step=1))
plt.grid(linestyle='-', linewidth=0.5)
plt.show()

# Silhouette Coefficient Method: visualisation silhouette scores
plt.figure(figsize=(7.5,5))
plt.plot(list(sil_coeff.keys()), list(sil_coeff.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.xticks(np.arange(2, 21, step=1))
plt.grid(linestyle='-', linewidth=0.5)
plt.show()


#%% Comparison of preprocessing techniques for KMeans clustering
 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

# KMeans clustering without preprocessing
# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]
n_clusters = 5
# initialise KMeans and fit data
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df)
# get clusters labels and assign them to dataframe
labels = kmeans.predict(df)
df['Labels']=labels
# visualisation
plt.figure(figsize=(16,16))
plt.subplot(221)
plt.title('No preprocessing')
cmap=sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with StandardScaler
from sklearn.preprocessing import StandardScaler
# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]
n_clusters = 5
# initialise KMeans and fit data
scaler = StandardScaler()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(df)
# get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels']=labels
# visualisation
plt.subplot(222)
plt.title('StandardScaler')
cmap=sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with Normalizer

from sklearn.preprocessing import Normalizer
# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]
# initialise KMeans and fit data
norm = Normalizer()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(norm, kmeans)
pipeline.fit(df)
# get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels']=labels
# visualisation
plt.subplot(223)
plt.title('Normalizer')
cmap=sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# KMeans clustering with MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler
# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]
# initialise KMeans and fit data
maxabs = MaxAbsScaler()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(maxabs, kmeans)
pipeline.fit(df)
# get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels']=labels
# visualisation
plt.subplot(224)
plt.title('MaxAbsScaler')
cmap=sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)


#%% Machine Learning: KMeans clustering with StandardScaler and visualisation in Seaborn
 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# define dataframe to apply the KMeans algorithm
df=red_stats.loc[:,['PTS','AST','TRB','STL','BLK', 'FG%','3P','3PA','3P%','2P',
                    '2PA','2P%','eFG%','USG%']]

# initialise KMeans and fit data
n_clusters = 5
scaler = StandardScaler()
kmeans = KMeans(n_clusters=n_clusters)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(df)

# get clusters labels and assign them to dataframe
labels = pipeline.predict(df)
df['Labels']=labels

cmap=sns.color_palette(palette="muted", n_colors=n_clusters)
sns.scatterplot(x='PTS', y='USG%', data=df, hue='Labels', palette=cmap)

# plot main stats in a pair plot after clustering
sns.pairplot(df, vars=['PTS','USG%','TRB','AST'], hue='Labels', palette="muted")

#%% Visualisation in Bokeh after KMeans
    
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import LabelSet, Title, HoverTool, CategoricalColorMapper
import pandas as pd
from bokeh.palettes import Category10

# define source for Bokeh graph
source = ColumnDataSource(data=dict(USG=list(red_stats['USG%']),
                                    PTS=list(red_stats['PTS']),
                                    AST=list(red_stats['AST']),
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
        factors=[str(i+1) for i in range(n_clusters)],
        palette=Category10[n_clusters])

# define and show graph USG% vs PTS
plot = figure(plot_width=1000, plot_height=400, tools=[hover])
plot.circle('USG', 'PTS', source=source, size=10, alpha=0.75, 
            color={'field': 'labels',
                   'transform': mapper})
plot.xaxis.axis_label = 'Usage %'
plot.yaxis.axis_label = 'Points'
output_file('USGvPTS.html')
show(plot)


# define a hover as player and season
hover2 = HoverTool(tooltips=[
        ('Player', '@desc'),
        ('Season', '@season'),
        ])
# define the colors for mapping the labels from KMeans
mapper2 = CategoricalColorMapper(
        factors=[str(i+1) for i in range(n_clusters)],
        palette=Category10[n_clusters])
# define source for Bokeh graph
source2 = ColumnDataSource(data=dict(USG=list(red_stats['USG%']),
                                    PTS=list(red_stats['PTS']),
                                    AST=list(red_stats['AST']),
                                    desc=list(red_stats['Player']),
                          
                            season=list(red_stats['Season']),
                                    labels=list(map(str, list(labels)))
                                    ))
# define and show graph PTS vs AST
plot2 = figure(plot_width=1000, plot_height=400, tools=[hover2])
plot2.circle('AST', 'PTS', source=source2, size=10, alpha=0.75, 
            color={'field': 'labels',
                   'transform': mapper2})
plot2.xaxis.axis_label = 'Assists'
plot2.yaxis.axis_label = 'Points'
output_file('ASTvPTS.html')
show(plot2)


# define a hover as player and season
hover3 = HoverTool(tooltips=[
        ('Player', '@desc'),
        ('Season', '@season'),
        ])
# define the colors for mapping the labels from KMeans
mapper3 = CategoricalColorMapper(
        factors=[str(i+1) for i in range(n_clusters)],
        palette=Category10[n_clusters])
# define source for Bokeh graph
source3 = ColumnDataSource(data=dict(USG=list(red_stats['USG%']),
                                    PTS=list(red_stats['PTS']),
                                    TRB=list(red_stats['TRB']),
                                    AST=list(red_stats['AST']),
                                    desc=list(red_stats['Player']),
                                    season=list(red_stats['Season']),
                                    labels=list(map(str, list(labels)))
                                    ))
# define and show graph PTS vs AST
plot3 = figure(plot_width=1000, plot_height=400, tools=[hover3])
plot3.circle('TRB', 'PTS', source=source3, size=10, alpha=0.75, 
            color={'field': 'labels',
                   'transform': mapper3})
plot3.xaxis.axis_label = 'Total Rebounds'
plot3.yaxis.axis_label = 'Points'
output_file('TRBvPTS.html')
show(plot3)


#%% Scree Plot for PCA 

from sklearn.decomposition import PCA

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component '+str(e) for e in range(1,11)])

# scree plot to measure the weight of each principal component
scree = pd.DataFrame({'Variation':pca.explained_variance_ratio_,
             'Principal Component':['PC'+str(e) for e in range(1,11)]})
sns.barplot(x='Principal Component',y='Variation', 
           data=scree, color="c")
plt.title('Scree Plot')
# PC1 explains more than 75% of the variation
# PC1 and PC2 together account for almost 90% of the variation 
# Using PC1 and PC2 would be a good approximation

#%% Machine Learning: PCA 2D visualisation

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
labelsNoIndex=list(df.Labels)
finalDf= principalDf
finalDf['labels'] = labels

fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

label=[str(i+1) for i in range(n_clusters)]
colors = Category10[n_clusters]
for label, color in zip(label,colors):
    indicesToKeep = finalDf['labels'] == label
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

# define a hover as player and season
hoverPCA = HoverTool(tooltips=[
        ('Player', '@desc'),
        ('Season', '@season'),
        ])
# define the colors for mapping the labels from KMeans
mapperPCA = CategoricalColorMapper(
        factors=[str(i+1) for i in range(n_clusters)],
        palette=Category10[n_clusters])
# define source for Bokeh graph
sourcePCA = ColumnDataSource(data=dict(x=list(finalDf['principal component 1']),
                                    y=list(finalDf['principal component 2']),
                                    desc=list(red_stats['Player']),
                                    season=list(red_stats['Season']),
                                    labels=list(map(str, list(labels)))
                                    ))
# define and show graph PC1 vs PC2
plotPCA = figure(plot_width=1000, plot_height=400, tools=[hoverPCA])
plotPCA.circle('x', 'y', source=sourcePCA, size=10, alpha=0.75, 
            color={'field': 'labels',
                   'transform': mapperPCA})
plotPCA.xaxis.axis_label = 'PC1'
plotPCA.yaxis.axis_label = 'PC2'
output_file('PCA.html')
show(plotPCA)

  
#%% PCA and comparison with Logistic Regression classifier

#https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train_std = sc.fit_transform(X_train)
#X_test_std = sc.transform(X_test)
#
#from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
#
## intialize pca and logistic regression model
#pca = PCA(n_components=2)
#lr = LogisticRegression(multi_class='auto', solver='liblinear')
#
## fit and transform data
#X_train_pca = pca.fit_transform(X_train_std)
#X_test_pca = pca.transform(X_test_std)
#lr.fit(X_train_pca, y_train)




