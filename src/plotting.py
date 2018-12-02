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
import random

# sns.set_style('whitegrid')
# sns.set_style("dark")
# data = wrangle_features(DATA)
# DEPENDENT = "shot_made_flag"


cNoFocus = '#000000'
cPlayoffs = '#42EFEF'#'#10A2A2'
cSeason = '#EF4242'#'#A21010'
cAll = '#9942EF'
cMisc = '#99EF42'
CP = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_style("whitegrid")
# data.shot_made_flag = data.shot_made_flag.astype('category')
# target_feature = DEPENDENT

def random_color():
    return random.choice(CP)

def corr_matrix(data: pd.DataFrame):
    """ Plot correlation_matrix for data """
    d = data.select_dtypes(np.number)
    f, g = plt.subplots(figsize=(8, 8))
    corr_matrix = np.corrcoef(d.T)
    g = smg.plot_corr(corr_matrix, xnames=d.columns)
    g.axes[0].set_title('Correlation Matrix', color = cNoFocus)

    for axis in g.axes:
        axis.tick_params(colors=cNoFocus)
        axis.spines['bottom'].set_color(cNoFocus)
        axis.spines['top'].set_color(cNoFocus)
        axis.spines['left'].set_color(cNoFocus)
        axis.spines['right'].set_color(cNoFocus)
        axis.set_xticklabels(axis.get_xticklabels(), size = 'small')
        axis.set_yticklabels(axis.get_yticklabels(), size = 'small')
    f.savefig(f"figs/corr_matrix.png", bbox_inches='tight', transparent = True, dpi = 200)
    # plt.close(f)
    return g


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
    f, g = plt.subplots(figsize=(8, 8))
    data = model.train_x.join(model.train_y)
    playoff_mask = data.playoffs == 1

    g = sns.regplot(
        x=data[playoff_mask].shot_distance,
        y=data[playoff_mask].shot_made_flag,
        logistic=True,
        scatter_kws = {
            'color': '#95a5a6'
        },
        line_kws = {
            'color': random_color(),
            'label': 'Playoffs',
            # 'alpha': 0.5
        }
        )
    g = sns.regplot(
        x=data[~playoff_mask].shot_distance,
        y=data[~playoff_mask].shot_made_flag,
        logistic=True,
        scatter_kws = {
            'color': '#95a5a6'
        },
        line_kws = {
            'color': random_color(),
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
    f.savefig(f"figs/plot_regular_vs_post_season.png", bbox_inches='tight', transparent = True, dpi = 200)
    plt.close(f)
    return g


def plot_proba(model):
    data = model.train_x.join(model.train_y)
    f, g = plt.subplots(figsize=(8, 8))
    cp = random_color()
    g = sns.regplot(
        x=data.shot_distance,
        y=data.shot_made_flag,
        logistic=True,
        scatter_kws = {
            'color': cp,
        },
        line_kws = {
            'color': cp,
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
    f.savefig(f"figs/plot_proba.png", bbox_inches='tight', transparent = True, dpi = 200)
    plt.close(f)
    return g

def plot_confusion_matrix(model):
    f, g = plt.subplots(figsize=(8, 8))
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
    f.savefig(f"figs/plot_confusion_matrix.png", bbox_inches='tight', transparent = True, dpi = 200)
    plt.close(f)
    return g

# https://www.kaggle.com/khozzy/kobe-shots-show-me-your-best-model/notebook

def plot_feature_boxplot(x: str = None, y: str = None, data: pd.DataFrame = None, ax = None):

    g = sns.boxplot(
                y= y,
                x= x,
                data=data,
                showmeans=True,
                ax = ax,
                color = random.choice(CP)
                )

    g.set_title(f'Boxplot of {y} and {x}')
    # g.set_xlabel(f'{x}', size = 'xx-large', color = cNoFocus)
    # g.set_ylabel(f'{y}', size = 'xx-large', color = cNoFocus)
    # g.set_xticklabels(g.get_xticklabels(), size = 'xx-large')
    # g.set_yticklabels(g.get_yticklabels(), size = 'xx-large')
    # g.tick_params(colors=cNoFocus)
    # g.spines['bottom'].set_color(cNoFocus)
    # g.spines['top'].set_color(cNoFocus)
    # g.spines['left'].set_color(cNoFocus)
    # g.spines['right'].set_color(cNoFocus)
    # g.xaxis.label.set_color(cNoFocus)
    # g.yaxis.label.set_color(cNoFocus)

    plt.tight_layout()
    # plt.show()

    return g

def plot_feature_boxplots(data):
    for feature in data.columns.tolist():
        f, ax = plt.subplots(figsize=(8, 8))
        g = plot_feature_boxplot(x = DEPENDENT, y = feature, data = data, ax = ax)
        f.savefig(f"figs/boxplot_{feature}.png", bbox_inches='tight', transparent = True, dpi = 200)
        plt.close(f)

def plot_feature_distplot(y: str = None, data: pd.DataFrame = None, ax = None):

    g = sns.distplot(data[y], norm_hist = False, label = y, color = random.choice(CP),
            kde_kws={"lw": 2, "label": "kde", "shade": False, "cumulative": False},
            hist_kws={"linewidth": 2}, ax = ax
            )

    g.set_title(f'Distribution of {y}', color = cNoFocus)
    g.set_xlabel(f'{y}', size = 'xx-large', color = cNoFocus)
    g.set_ylabel(f'Density', size = 'xx-large', color = cNoFocus)
    plt.tight_layout()
    return g

def plot_feature_distplots(data):
    for feature in data.select_dtypes(include = [np.number]).columns.tolist():
        f, g = plt.subplots(figsize=(8, 8))
        g = plot_feature_distplot(y = feature, data = data, ax = g)
        f.savefig(f"figs/distplot_{feature}.png", bbox_inches='tight', transparent = True, dpi = 200)
        plt.close(f)


model = model2
y_score = model.predict_labels(model.test_x)

def plot_roc(model, ax = None):
    y_score = model.predict_labels(model.test_x)
    fpr, tpr, _ = roc_curve(model.test_y, y_score)

    roc_auc = auc(fpr, tpr)

    f, ax = plt.subplots(figsize=(8, 8))
    g = sns.lineplot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (1, roc_auc), ax = ax)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example', color = cNoFocus)
    plt.legend(loc="lower right")
    g.set_xticklabels(g.get_xticklabels(), size = 'xx-large')
    g.set_yticklabels(g.get_yticklabels(), size = 'xx-large')
    g.tick_params(colors=cNoFocus)
    g.spines['bottom'].set_color(cNoFocus)
    g.spines['top'].set_color(cNoFocus)
    g.spines['left'].set_color(cNoFocus)
    g.spines['right'].set_color(cNoFocus)
    g.xaxis.label.set_color(cNoFocus)
    g.yaxis.label.set_color(cNoFocus)
    f.savefig(f"figs/roc_plot.png", bbox_inches='tight', transparent = True, dpi = 200)
    plt.close(f)
    return g

# sns.palplot(CP)
plot_feature_distplots(data)
plot_feature_boxplots(data)
data = DATA
x = DEPENDENT
y = 'seconds_remaining'



def quantile_plot(x, **kwargs):
    qntls, xr = stats.probplot(x, fit=False)
    g = plt.scatter(xr, qntls, **kwargs)
    return g

def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)


def plot_feature_qqplot(y: str = None, data: pd.DataFrame = None, ax = None):

    g = sns.distplot(data[y], norm_hist = False, label = y, color = random.choice(CP),
            kde_kws={"lw": 2, "label": "kde", "shade": False, "cumulative": False},
            hist_kws={"linewidth": 2}, ax = ax
            )

    g.set_title(f'Distribution of {y}', color = cNoFocus)
    g.set_xlabel(f'{y}', size = 'xx-large', color = cNoFocus)
    g.set_ylabel(f'Density', size = 'xx-large', color = cNoFocus)
    plt.tight_layout()
    return g

def plot_feature_qqplot(data):
    for feature in data.select_dtypes(include = [np.number]).columns.tolist():
        f, g = plt.subplots(figsize=(8, 8))
        g = plot_feature_distplot(y = feature, data = data, ax = g)
        f.savefig(f"figs/distplot_{feature}.png", bbox_inches='tight', transparent = True, dpi = 200)
        plt.close(f)


g = sns.FacetGrid(data, col="playoffs", height=4)

quantile_plot(data.shot_distance)

g = sns.PairGrid(data)
g.map(qqplot)




# sns.kdeplot(data[y])

sns.countplot(shots_last10seconds, hue = indicator)

sns.set()
sns.boxplot(
            y= y,
            x= x,
            data=data,
            showmeans=True,
            # ax = ax
            )


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
