
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.linear_model import LogisticRegression

import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import (lag_plot,
							autocorrelation_plot,
							table, scatter_matrix,
							boxplot)

from patsy import dmatrices
from math import degrees, acos
from scipy.spatial import distance


# os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.getcwd()+'/src')
from eda import *
from confusion_matrix_pretty import *
# from plotting import *
from logistic_regression import *
from linear_discriminant_analysis import *

def cols(df: pd.DataFrame) -> list:
    """Extract list of columns from input DataFrame and removing the dependent variable."""

    return [x for x in df.columns.tolist() if x not in [DEPENDENT]]

def get_dummies(df: pd.DataFrame, drop_first = False):
    """Replace catagorical varables with indicators"""

    df = pd.get_dummies(df, dtype = float)
    df.columns = df.columns \
                    .str.lower() \
                    .str.replace(" ", "_")
    return df

def LogRegModel(data: pd.DataFrame, add_constant = False):
    if add_constant:
        data = sm.add_constant(data)
    model = LogR(data, DEPENDENT)
    model.sm = model.statsmodel()

    model.yhat = model.sm.predict(model.test_x)
    print("\n Predicted Log Loss: {}\n".format(
        round(
            log_loss(model.test_y, model.yhat)
            , 4)))
    return model

def summarize_model(model: LogR):
    print(model.describe_features())
    print(model.sm.summary())
    print(model.sm.summary2())
    print(model.sm.wald_test_terms())

# Import Data
DATA = pd.read_excel('data/project2Data.xlsx', index_col = 'recId')
FOR_PREDICTION = pd.read_excel('data/project2Pred.xlsx', index_col = 'rannum')
DEPENDENT = "shot_made_flag"

REDUNDANT_FEATURES = [
		'team_id', # constant term
		'team_name', # constant term
		'season',
		'game_id', # violates independence
		'matchup',
		'shot_id',
		'recId',
		'shot_zone_area',
		'shot_zone_basic',
		'shot_zone_range',
		'minutes_remaining',
		'seconds_elapsed_in_game',
		'game_event_id',  # violates independence
		# 'game_date',
		'action_type',
        'loc_x', # collinear with lat
        'loc_y', # collinear with lon

	]




"""############# Model 0 - Predicted Log Loss: 0.6552 #############"""


"""Dataset: d0 | Prediction set: d0_pred
    - Full Model
"""

d0 = prepare_data(DATA.drop(columns = ['action_type']))
d0.game_date = d0.game_date.apply(lambda x: x.toordinal())
d0 = get_dummies(d0).fillna(0) # Get dummy variables for categoricals
d0_pred = wrangle_features(FOR_PREDICTION)
d0_pred.game_date = d0_pred.game_date.apply(lambda x: x.toordinal())
d0_pred = get_dummies(d0_pred).fillna(0) # Get dummy variables for categoricals
d0_pred = d0_pred[cols(d0)]

"""Fit d2"""

model0 = LogRegModel(d0)
summarize_model(model0)
model0.roc_plot()

"""############# Model 1 - Predicted Log Loss: 0.6652 #############"""


"""Dataset: d1 | Prediction set: d1_pred
    - No categorical features
"""
d1 = prepare_data(DATA, drop_categorical = True) # Wrangle Data
d1.game_date = d1.game_date.apply(lambda x: x.toordinal())
d1 = d1.fillna(0)
# d1_pred = wrangle_features(FOR_PREDICTION)
# d1_pred.game_date = d1_pred.game_date.apply(lambda x: x.toordinal())
# d1_pred = d1_pred[cols(d1)].fillna(0)

"""Fit d1"""
model1 = LogRegModel(d1)
summarize_model(model1)
model1.roc_plot()

"""############# Model 2 - Predicted Log Loss: 0.6479 #############"""

"""Dataset: d2 | Prediction set: d2_pred
    - Categorical features as indicators
    - Drop redundant features
"""

d2 = prepare_data(DATA, drop_columns= REDUNDANT_FEATURES)
d2.game_date = d2.game_date.apply(lambda x: x.toordinal())
# d2.last_seconds_of_period = d2.last_seconds_of_period.astype(int)
d2 = get_dummies(d2).fillna(0) # Get dummy variables for categoricals
# d2_pred = wrangle_features(FOR_PREDICTION)
# d2_pred = get_dummies(d2_pred).fillna(0) # Get dummy variables for categoricals
# d2_pred = d2_pred[cols(d2)]

"""Fit d2"""
model2 = LogRegModel(d2)
summarize_model(model2)
model2.roc_plot()
pairplot(x = d2)

odds = np.exp(model2.sm.params).sort_values(ascending = False)

""" Refine Model 2 """

wald = model2.sm.wald_test_terms()
wald.df = wald.summary_frame()
wald.significant = wald.df[wald.df['P>chi2'] < 0.1].index.tolist()

""" Refined Fit - Predicted Log Loss: 0.6634 """

model2r = LogRegModel(d2[wald.significant + [DEPENDENT]])

"""
lat                      -0.1393
shot_distance            -0.0447
attendance                0.0002
arena_temp                0.0337
seconds_left_in_game      0.0001
last_seconds_of_period   -0.8275
"""


#! Interpret: http://www-hsc.usc.edu/~eckel/biostat2/notes/notes14.pdf

"""############# Model 3 - Predicted Log Loss: 0.669 #############"""

"""Dataset: d3 | Prediction set: d3_pred
    - Allen's Model
"""
d3cols = [
    'shot_distance',
    'playoffs',
    'arena_temp',
    'game_event_id',
    'lat',
    'lon',
    'shot_made_flag'
    ]
d3 = DATA[d3cols]
d3_pred = FOR_PREDICTION[d3cols].drop(columns = [DEPENDENT]).fillna(0)

"""Fit d3"""
model3 = LogRegModel(d3)
summarize_model(model3)


"""############# Model 4 - Predicted Log Loss: ??? #############"""


"""Dataset: d4 | Prediction set: d1_pred
    - No categorical features
"""
d4 = prepare_data(DATA) # Wrangle Data
d4.game_date = d4.game_date.apply(lambda x: x.toordinal())
d4 = d4.fillna(0)
d4_pred = wrangle_features(FOR_PREDICTION)
d4_pred.game_date = d4_pred.game_date.apply(lambda x: x.toordinal())
d4_pred = d4_pred[cols(d4)].fillna(0)

"""Fit d1"""
model1 = LogRegModel(d1)
summarize_model(model1)
model1.roc_plot()













#! Data Overview

# TODO: Add Univariate Plots
    # QQ
    # Hist

sm.qqplot(DATA.arena_temp, stats.t, fit=True, line='45')

#? Correlation Matrix





"""
•	The __odds of Kobe making a shot decrease with respect to the distance he is from the hoop__.  If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

•	The __probability of Kobe making a shot decreases linearly with respect to the distance he is from the hoop__.    If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

•	The relationship between the __distance Kobe is from the basket and the odds of him making the shot is different if they are in the playoffs__.  Quantify your findings with statistical evidence one way or the other. (Tests, CIs, plots, etc.) 
"""


""" Odds Ratios
Odds ratios that are greater than 1 indicate that the event is more likely to occur as the predictor increases. Odds ratios that are less than 1 indicate that the event is less likely to occur as the predictor increases.

https://www.predictiveanalyticsworld.com/patimes/on-variable-importance-in-logistic-regression/9649/



"""


# model = LogR(d1, DEPENDENT)
# model.describe_features()
# model.fit()
# model.score()
# model.predict(d1_pred)
# model.log_loss()
# model.confusion_matrix()
# model.roc_plot()


