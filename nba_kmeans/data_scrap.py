from nba_kmeans.stats_season import stats_season
from nba_kmeans.stats_season_adv import stats_season_adv

stats_bas = stats_season(2019)
stats_bas.to_csv('data/stats_bas.csv')
stats_adv = stats_season_adv(2019)
stats_adv.to_csv('data/stats_adv.csv')