
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
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
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
sys.path.insert(0, os.getcwd()+'/src')
from eda import *
from confusion_matrix_pretty import *
# from plotting import *
from logistic_regression import *
from linear_discriminant_analysis import *

# Define utility functions
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

def to_latex(df):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    with open('temp.txt', 'w') as f:
        f.write(r'\resizebox{\textwidth}{!}{'+
            df.to_latex()
            + r'}\captionof{table}{Feature Summary}\label{tbl:featuresummary}')
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

def desc(df: pd.DataFrame):
	"""Produces a summary of the input DataFrame

	Arguments:
		df {pd.DataFrame} -- [description]

	Returns:
		pd.DataFrame -- DataFrame of summary statistics
	"""

	desc = df.describe(percentiles = None).T
	desc['missing'] = len(df.index) - desc['count']
	# desc = desc.astype('int')
	desc['median'] = df.median()
	desc['missing %'] = desc.missing / len(df.index) * 100
	return desc.T


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
		'game_date', # violates independence
		'action_type',
        'loc_x', # collinear with lat
        'loc_y', # collinear with lon

	]



"""############# Model 2 - Predicted Log Loss: 0.6479 #############"""

"""Dataset: d2 | Prediction set: d2_pred
    - Categorical features as indicators
    - Drop redundant features
"""

d2 = prepare_data(DATA, drop_columns= REDUNDANT_FEATURES)
d2 = get_dummies(d2).fillna(0) # Get dummy variables for categoricals
d2_pred = wrangle_features(FOR_PREDICTION)
d2_pred = get_dummies(d2_pred).fillna(0) # Get dummy variables for categoricals
d2_pred = d2_pred[cols(d2)]

"""Fit d2"""
model2 = LogRegModel(d2)
summarize_model(model2)
model2.sm2 = model2.statsmodel_()
model2.sm2.fitted = model2.sm2.fit()
# model2.sm.summary2()

""" Predict Probabilities"""
model2.sm2.fitted.predict(model2.test_x)

""" Predict Labels """
pd.Series(model2.predict_labels(d2_pred)).set_index(d2_pred.shot_id)

""" Generate density curves """
pdf_train_x = model2.sm2.pdf(model2.train_x)
cdf_train_x = model2.sm2.cdf(model2.train_x)

""" Capture results and convert to odds and precentage change for interpretation """
result = model2.sm.summary2()
logodds = result.tables[1]
odds[['Coef.','[0.025', '0.975]']] = np.exp(logodds[['Coef.','[0.025', '0.975]']])
pct_change = (odds[['Coef.','[0.025', '0.975]']] - 1) * 100

""" Generate model plots using src/plotting.py"""
corr_matrix(wrangle_features(DATA.drop(columns = ['action_type'])))
plot_proba(model2)
plot_regular_vs_post_season(model2)
plot_confusion_matrix(model2)



