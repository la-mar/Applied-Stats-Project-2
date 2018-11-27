from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import statsmodels.graphics.api as smg
import sys
import os
from scipy.stats import norm

sns.set_style('whitegrid')

data = wrangle_features(DATA)
DEPENDENT = "shot_made_flag"


data.shot_made_flag = data.shot_made_flag.astype('category')
target_feature = DEPENDENT

def corr_matrix(data: pd.DataFrame):
    d = data.select_dtypes(np.number)
    corr_matrix = np.corrcoef(d.T)
    smg.plot_corr(corr_matrix, xnames=d.columns)


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


g = sns.boxplot(x="variable", y="value", data=pd.melt(d1))
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.tight_layout()


g = sns.FacetGrid(
            data,
            col='combined_shot_type',
            # row='shot_made_flag',
            hue='shot_made_flag',
            col_wrap = 5,
            height = 3
            )
g = g.map(sns.countplot,
        'season',
        orient = 'v'
        # bins = 25,
        # fit = norm,
        # kde_kws={"color": "red", "lw": 1, "label": "KDE"}
        )


temp.melt(id_vars = ['seconds_left_in_period']).groupby(['seconds_left_in_period', 'value']).count()

temp.set_index('seconds_left_in_period').pivot(columns = 'shot_made_flag')

# g.set_xticklabels( np.linspace(1, 10, num = 10))

# target_feature = data.shot_distance

# indicator = data.shot_made_flag

# sns.distplot(target_feature)


# d = data
# d['shots_last10sec']


#! Most outliers are at or near 0 secords left in the current period
g = sns.countplot(
    x = 'seconds_left_in_period',
    # y = 'shot_made_flag',
    hue = 'shot_made_flag',
    # col='period',
    data = data[data.seconds_left_in_period <= 15]
    )

temp = data[['seconds_left_in_period', 'shot_made_flag']][data.seconds_left_in_period <= 15]
temp['shot_made_%'] =

g = sns.countplot(
    x = 'seconds_left_in_period',
    # y = 'shot_made_flag',
    hue = 'shot_made_flag',
    # col='period',
    data = temp
    )

#! Made/Missed by combined_action_type by season

# sns.boxplot(x = indicator, y = shots_last10seconds)

sns.distplot(data.seconds_left_in_period, bins = 10)
# sns.countplot(shots_last10seconds, hue = indicator)


"""
lat                             float64
lon                             float64
playoffs                          int64
seconds_remaining                 int64
shot_distance                     int64
shot_made_flag                    int64
game_date                         int64
attendance                        int64
arena_temp                        int64
avgnoisedb                      float64
seconds_left_in_period            int64
seconds_left_in_game              int64
home_or_away                      int32
num_shots_cumulative              int64
angle_from_basket               float64
season_count                       int8
last_seconds_of_period            int32
"""

# https://www.kaggle.com/khozzy/kobe-shots-show-me-your-best-model/notebook
f, axarr = plt.subplots(4, 2, figsize=(15, 15))

sns.boxplot(
    x='lat',
    y='shot_made_flag',
    data=data,
    showmeans=True,
    ax=axarr[0,0]
    )
sns.boxplot(
    x='lon',
    y='shot_made_flag',
    data=data,
    showmeans=True,
    ax=axarr[0, 1]
    )
sns.boxplot(
    x='loc_y',
    y='shot_made_flag',
    data=data,
    showmeans=True,
    ax=axarr[1, 0]
    )
sns.boxplot(
    x='loc_x',
    y='shot_made_flag',
    data=data,
    showmeans=True,
    ax=axarr[1, 1]
    )
sns.boxplot(
    x='minutes_remaining',
    y='shot_made_flag',
    showmeans=True,
    data=data,
    ax=axarr[2, 0]
    )
sns.boxplot(
    x='seconds_remaining',
    y='shot_made_flag',
    showmeans=True,
    data=data,
    ax=axarr[2, 1]
    )
sns.boxplot(
    x='shot_distance',
    y='shot_made_flag',
    data=data,
    showmeans=True,
    ax=axarr[3, 0]
    )

axarr[0, 0].set_title('Latitude')
axarr[0, 1].set_title('Longitude')
axarr[1, 0].set_title('Loc y')
axarr[1, 1].set_title('Loc x')
axarr[2, 0].set_title('Minutes remaining')
axarr[2, 1].set_title('Seconds remaining')
axarr[3, 0].set_title('Shot distance')

plt.tight_layout()
plt.show()





def pairplot(x: pd.DataFrame = None, model = None):

    if model is not None:
        data = model.train_x
    else:
        data = x

    if DEPENDENT in data.columns.tolist():
        d = data[DEPENDENT].astype('category')

    data = data.select_dtypes(exclude = ['object', 'category'])
    data[DEPENDENT] = d

    sns.pairplot(data, vars=data.columns.tolist(), hue=DEPENDENT, size=3)
    plt.show()


def plot_redundants():
    sns.pairplot(
        data[REDUNDANT_FEATURES + [DEPENDENT]],
        # vars=['loc_x', 'loc_y', 'lat', 'lon', 'shot_distance'],
        hue=DEPENDENT,
        size=3)
    plt.show()




"""
recId                               int64
action_type                      category
combined_shot_type               category
game_event_id                       int64
game_id                             int64
lat                               float64
loc_x                               int64
loc_y                               int64
lon                               float64
minutes_remaining                   int64
period                              int64
playoffs                            int64
season                           category
seconds_remaining                   int64
shot_distance                       int64
shot_made_flag                      int64
shot_type                        category
shot_zone_area                   category
shot_zone_basic                  category
shot_zone_range                  category
team_id                             int64
team_name                        category
game_date                  datetime64[ns]
matchup                          category
opponent                         category
shot_id                             int64
attendance                          int64
arena_temp                          int64
avgnoisedb                        float64
seconds_left_in_period              int64
seconds_elapsed_in_game             int64
seconds_left_in_game                int64
home_or_away                        int32
num_shots_cumulative                int64
angle_from_basket                 float64
season_count                         int8
dtype: object
"""
