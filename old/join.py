# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:46:21 2020

@author: user
"""



#%% Join tables

from stats_season import stats_season
from stats_season_adv import stats_season_adv
import pandas as pd

stats_bas = stats_season(2019, 2020)
stats_adv = stats_season_adv(2019, 2020)

stats = pd.concat([stats_bas, stats_adv], axis=1)
stats = stats.loc[:,~stats.columns.duplicated()]

#%% Machine Learning: KMeans clustering
 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
   
df=stats_adv.iloc[:,5:]

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(df)
labels = model.predict(df)
print(model.inertia_)

df['labels']=labels

sns.pairplot(df, vars=['PTS','USG%','TRB','AST'], hue='labels', palette="husl")

#%% Machine Learning: clustering

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=3)
model.fit(df)
labels = model.predict(df)
print(model.inertia_)

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


