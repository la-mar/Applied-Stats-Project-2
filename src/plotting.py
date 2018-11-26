from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

import sys
import os



def pca_bin_plot(pca_result: pd.DataFrame, train_y: pd.DataFrame) -> None:
    """Creates a scatterplot of binary PCA results.
    
    Arguments:
        pca_result {pd.DataFrame} -- (n_obs, n_components)
        train_y {pd.DataFrame} -- depended variable
    
    Returns:
        None
    """

    plt.figure()

    plt.scatter(
        pca_result[train_y == 0].PC1, 
        pca_result[train_y == 0].PC2, 
        color='red', 
        alpha=.25, 
        lw=1,
        label='shot_missed'
        )

    plt.scatter(
        pca_result[train_y == 1].PC1, 
        pca_result[train_y == 1].PC2, 
        color='green', 
        alpha=.25, 
        lw=1,
        label='shot_made'
        )

    def draw_vector(v0, v1, ax=None):
        ax = ax or plt.gca()
        arrowprops=dict(arrowstyle='->',
                        linewidth=2,
                        shrinkA=0, shrinkB=0)
        ax.annotate('', v1, v0, arrowprops=arrowprops)

    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal');

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Shots')

    plt.show()

def stacked_bar_plot(x: pd.Series, indicator: pd.Series) -> None:
    """Plots a stacked bar chart of x, colored by the indicator variable.
    
    Arguments:
        x {pd.Series} -- [description]
        indicator {pd.Series} -- [description]
    
    Returns:
        None -- [description]
    """

    pd.crosstab(x, indicator).plot.bar(stacked=True)
    plt.legend(title=indicator.name)

# def pairgrid_reg():
#     g = sns.PairGrid(temp_x[['shot_made_flag', 'shot_distance']], hue="shot_made_flag") 
#     g.map_upper(sns.regplot) 
#     g.map_lower(sns.residplot) 
#     g.map_diag(plt.hist) 
#     for ax in g.axes.flat: 
#         plt.setp(ax.get_xticklabels(), rotation=45) 
#     g.add_legend() 
#     g.set(alpha=0.5)



sns.boxplot()


[
 'recId',
 'action_type',
 'combined_shot_type',
 'game_event_id',
 'game_id',
 'lat',
 'loc_x',
 'loc_y',
 'lon',
 'minutes_remaining',
 'period',
 'playoffs',
 'season',
 'seconds_remaining',
 'shot_distance',
 'shot_made_flag',
 'shot_type',
 'shot_zone_area',
 'shot_zone_basic',
 'shot_zone_range',
 'team_id',
 'team_name',
 'game_date',
 'matchup',
 'opponent',
 'shot_id',
 'attendance',
 'arena_temp',
 'avgnoisedb',
 'seconds_left_in_period',
 'home_or_away',
 'num_shots_cumulative',
 'angle_from_basket',
 'season_count',
 'seconds_elapsed_in_game',
 'seconds_left_in_game']


sns.boxplot(x="variable", y="value", data=pd.melt(d1))





# target_feature = data.shot_distance

# indicator = data.shot_made_flag

# sns.distplot(target_feature)


# shots_last10seconds = data[data.seconds_left_in_period < 10].seconds_left_in_period

# # Most outliers are at or near 0 secords left in the current period
# g = (sns.jointplot(target_feature, data.seconds_left_in_period,color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6))

# sns.boxplot(x = indicator, y = shots_last10seconds)

# sns.distplot(shots_last10seconds, bins = 10)
# sns.countplot(shots_last10seconds, hue = indicator)









