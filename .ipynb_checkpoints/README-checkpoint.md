# NBA Players KMeans clustering

## Description
Extraction of NBA statistics from "basketball-reference.com" to visualise them and detect trends and patterns. 
Clustering of the top players using KMeans algorithm. 

## Revised skills
- Web scrapping with BeautifulSoup
- Data cleaning with pandas
- Visualisation in Seaborn and Bokeh 
- Unsupervised learning: KMeans clustering and PCA 2D visualisation with scikit-learn

## Results
Visualisation of the PCA components 1 and 2 in Bokeh after KMeans clustering:
![Clustering](/images/PCA_clustering.png)

Out of this analysis it can be concluded that there are 4 kinds of NBA players:
- Blue cluster: secondary players with low importance on the offense
- Green cluster: secondary players in their respective teams
- Grey cluster: athletic players with high scores, stars in their teams
- Yellow cluster: defensive specialists