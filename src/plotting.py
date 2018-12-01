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

# sns.set_style('whitegrid')
# sns.set_style("dark")
# data = wrangle_features(DATA)
# DEPENDENT = "shot_made_flag"


cNoFocus = '#ffffff'#'#9C9C9C'
cPlayoffs = '#42EFEF'#'#10A2A2'
cSeason = '#EF4242'#'#A21010'
cAll = '#9942EF'
cMisc = '#99EF42'
# data.shot_made_flag = data.shot_made_flag.astype('category')
# target_feature = DEPENDENT

def corr_matrix(data: pd.DataFrame):
    """ Plot correlation_matrix for data """
    d = data.select_dtypes(np.number)
    corr_matrix = np.corrcoef(d.T)
    g = smg.plot_corr(corr_matrix, xnames=d.columns)
    g.axes[0].set_title('Correlation Matrix', color = cNoFocus)

    for axis in g.axes:
        axis.tick_params(colors=cNoFocus)
        axis.spines['bottom'].set_color(cNoFocus)
        axis.spines['top'].set_color(cNoFocus)
        axis.spines['left'].set_color(cNoFocus)
        axis.spines['right'].set_color(cNoFocus)
        axis.set_xticklabels(axis.get_xticklabels(), size = 'xx-large')
        axis.set_yticklabels(axis.get_yticklabels(), size = 'xx-large')


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


# data.shot_made_flag = data.shot_made_flag.astype('category')


def plot_regular_vs_post_season(model):
    data = model.train_x.join(model.train_y)
    playoff_mask = data.playoffs == 1

    cNoFocus = '#ffffff'#'#9C9C9C'
    cPlayoffs = '#42EFEF'#'#10A2A2'
    cSeason = '#EF4242'#'#A21010'
    g = sns.regplot(
        x=data[playoff_mask].shot_distance,
        y=data[playoff_mask].shot_made_flag,
        logistic=True,
        scatter_kws = {
            'color': cNoFocus
        },
        line_kws = {
            'color': cPlayoffs,
            'label': 'Playoffs',
            # 'alpha': 0.5
        }
        )
    g = sns.regplot(
        x=data[~playoff_mask].shot_distance,
        y=data[~playoff_mask].shot_made_flag,
        logistic=True,
        scatter_kws = {
            'color': cNoFocus
        },
        line_kws = {
            'color': cSeason,
            'label': 'Regular Season',
            # 'alpha': 0.5
        }
        )
    g.set_xlim(0, 80)
    g.set_yticks([0, 0.25, 0.50, 0.75, 1])
    g.set_yticklabels(labels = ['Missed', 0.25, 0.50, 0.75,  'Made'], size = 'large')
    g.axvline(
        x=47,
        color=cNoFocus,
        linestyle=':',
        lw = 1,
        )
    g.legend()
    g.set_title('Regular and Post Season Shot Distance', color = cNoFocus)
    g.set_xlabel('Shot Distance (ft)', size = 'large', color = cNoFocus)
    g.set_ylabel('Shot Made/Missed', size = 'large', color = cNoFocus)
    g.tick_params( colors=cNoFocus)
    g.spines['bottom'].set_color(cNoFocus)
    g.spines['top'].set_color(cNoFocus)
    g.spines['left'].set_color(cNoFocus)
    g.spines['right'].set_color(cNoFocus)
    g.xaxis.label.set_color(cNoFocus)
    g.yaxis.label.set_color(cNoFocus)


def plot_proba(model):
    data = model.train_x.join(model.train_y)

    g = sns.regplot(
        x=data.shot_distance,
        y=data.shot_made_flag,
        logistic=True,
        scatter_kws = {
            'color': '#7332B3',
        },
        line_kws = {
            'color': cAll,
            'label': 'Shot Distance',
            # 'alpha': 0.5
        }
        )
    g.set_xlim(0, 80)
    g.set_yticks([0, 0.25, 0.50, 0.75, 1])
    g.set_yticklabels(labels = ['Missed', 0.25, 0.50, 0.75,  'Made'], size = 'large')
    g.axvline(
        x=47,
        color=cNoFocus,
        linestyle=':',
        lw = 1,
        )
    g.axvline(
        x=23.9,
        color=cNoFocus,
        linestyle=':',
        lw = 1,
        )
    g.legend()
    g.set_title('Predicted Probability of Shot Distance', color = cNoFocus)
    g.set_xlabel('Shot Distance (ft)', size = 'large', color = cNoFocus)
    g.set_ylabel('Shot Made/Missed', size = 'large', color = cNoFocus)
    g.tick_params( colors=cNoFocus)
    g.spines['bottom'].set_color(cNoFocus)
    g.spines['top'].set_color(cNoFocus)
    g.spines['left'].set_color(cNoFocus)
    g.spines['right'].set_color(cNoFocus)
    g.xaxis.label.set_color(cNoFocus)
    g.yaxis.label.set_color(cNoFocus)

def plot_confusion_matrix(model):

    cm = pd.DataFrame(model.sm.pred_table(),
                    index = [0, 1], columns = [0, 1])

    g = pretty_plot_confusion_matrix(cm, cmap='Oranges')
    g.set_title('Confusion Matrix', color = cNoFocus)
    g.set_xlabel('Actual', size = 'xx-large', color = cNoFocus)
    g.set_ylabel('Predicted', size = 'xx-large', color = cNoFocus)
    g.set_xticklabels(g.get_xticklabels(), size = 'xx-large')
    g.set_yticklabels(g.get_yticklabels(), size = 'xx-large')
    g.tick_params(colors=cNoFocus)
    g.spines['bottom'].set_color(cNoFocus)
    g.spines['top'].set_color(cNoFocus)
    g.spines['left'].set_color(cNoFocus)
    g.spines['right'].set_color(cNoFocus)
    g.xaxis.label.set_color(cNoFocus)
    g.yaxis.label.set_color(cNoFocus)
    return g

# https://www.kaggle.com/khozzy/kobe-shots-show-me-your-best-model/notebook

def plot_feature_boxplots(model):

    data = model.test_x.join(model.test_y)

    f, axarr = plt.subplots(6, 2, figsize=(15, 15))

    for idx, feature in enumerate(data.columns.sort_values().tolist()):
        sns.boxplot(
            y='lat',
            x='shot_made_flag',
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
        x='playoffs',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[1, 0]
        )
    sns.boxplot(
        x='last_seconds_of_period',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[1, 1]
        )
    sns.boxplot(
        x='seconds_left_in_period',
        y='shot_made_flag',
        showmeans=True,
        data=data,
        ax=axarr[2, 0]
        )
    sns.boxplot(
        x='avgnoisedb',
        y='shot_made_flag',
        showmeans=True,
        data=data,
        ax=axarr[2, 1]
        )
    sns.boxplot(
        x='arena_temp',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[3, 0]
        )
    sns.boxplot(
        x='attendance',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[3, 1]
        )
    sns.boxplot(
        x='shot_distance',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[4, 0]
        )
    sns.boxplot(
        x='home_or_away',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[4, 1]
        )
    sns.boxplot(
        x='num_shots_cumulative',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[5, 0]
        )
    sns.boxplot(
        x='angle_from_basket',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[5, 1]
        )
    sns.boxplot(
        x='season_count',
        y='shot_made_flag',
        data=data,
        showmeans=True,
        ax=axarr[6, 0]
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



if __name__ == "__main__":
    pass
    #! QQ plot
    # sm.qqplot(DATA.arena_temp, stats.t, fit=True, line='45')


    # g = sns.boxplot(x="variable", y="value", data=pd.melt(DATA))
    # g.set_xticklabels(g.get_xticklabels(), rotation=30)
    # plt.tight_layout()


    # data = model2.test_x.join(model2.test_y)

    # g = sns.FacetGrid(
    #             data,
    #             col='playoffs',
    #             # row='shot_made_flag',
    #             hue='playoffs',
    #             # col_wrap = 5,
    #             height = 3
    #             )
    # g = g.map(sns.regplot,
    #         x = 'shot_distance',
    #         y = 'shot_made_flag',
    #         data = data,
    #         logistic = True
    #         # bins = 25,
    #         # fit = norm,
    #         # kde_kws={"color": "red", "lw": 1, "label": "KDE"}
    #         )

    # # g.title = 'Regular and Post Season Shot Distance'

    # temp.melt(id_vars = ['seconds_left_in_period']).groupby(['seconds_left_in_period', 'value']).count()

    # temp.set_index('seconds_left_in_period').pivot(columns = 'shot_made_flag')

    # # g.set_xticklabels( np.linspace(1, 10, num = 10))

    # # target_feature = data.shot_distance

    # # indicator = data.shot_made_flag

    # # sns.distplot(target_feature)


    # # d = data
    # # d['shots_last10sec']


    # #! Most outliers are at or near 0 secords left in the current period
    # g = sns.countplot(
    #     x = 'seconds_left_in_period',
    #     # y = 'shot_made_flag',
    #     hue = 'shot_made_flag',
    #     # col='period',
    #     data = data[data.seconds_left_in_period <= 15]
    #     )

    # temp = data[['seconds_left_in_period', 'shot_made_flag']][data.seconds_left_in_period <= 15]
    # # temp['shot_made_%'] =

    # g = sns.countplot(
    #     x = 'seconds_left_in_period',
    #     # y = 'shot_made_flag',
    #     hue = 'shot_made_flag',
    #     # col='period',
    #     data = temp
    #     )

    #! Made/Missed by combined_action_type by season

    # sns.boxplot(x = indicator, y = shots_last10seconds)

    # sns.distplot(data.seconds_left_in_period, bins = 10)
    # sns.countplot(shots_last10seconds, hue = indicator)


#     """
#     lat                             float64
#     lon                             float64
#     playoffs                          int64
#     seconds_remaining                 int64
#     shot_distance                     int64
#     shot_made_flag                    int64
#     game_date                         int64
#     attendance                        int64
#     arena_temp                        int64
#     avgnoisedb                      float64
#     seconds_left_in_period            int64
#     seconds_left_in_game              int64
#     home_or_away                      int32
#     num_shots_cumulative              int64
#     angle_from_basket               float64
#     season_count                       int8
#     last_seconds_of_period            int32
#     """



# """
# recId                               int64
# action_type                      category
# combined_shot_type               category
# game_event_id                       int64
# game_id                             int64
# lat                               float64
# loc_x                               int64
# loc_y                               int64
# lon                               float64
# minutes_remaining                   int64
# period                              int64
# playoffs                            int64
# season                           category
# seconds_remaining                   int64
# shot_distance                       int64
# shot_made_flag                      int64
# shot_type                        category
# shot_zone_area                   category
# shot_zone_basic                  category
# shot_zone_range                  category
# team_id                             int64
# team_name                        category
# game_date                  datetime64[ns]
# matchup                          category
# opponent                         category
# shot_id                             int64
# attendance                          int64
# arena_temp                          int64
# avgnoisedb                        float64
# seconds_left_in_period              int64
# seconds_elapsed_in_game             int64
# seconds_left_in_game                int64
# home_or_away                        int32
# num_shots_cumulative                int64
# angle_from_basket                 float64
# season_count                         int8
# dtype: object
# """
