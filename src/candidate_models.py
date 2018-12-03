
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
# d2.game_date = d2.game_date.apply(lambda x: x.toordinal())
# d2.last_seconds_of_period = d2.last_seconds_of_period.astype(int)
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
model2.sm2.fitted.predict(model2.test_x)
# pd.Series(model2.predict_labels(d2_pred)).set_index(d2_pred.sho)
# model2.roc_plot()

pdf_train_x = model2.sm2.pdf(model2.train_x)
cdf_train_x = model2.sm2.cdf(model2.train_x)

result = model2.sm.summary2()
logodds = result.tables[1]
odds[['Coef.','[0.025', '0.975]']] = np.exp(logodds[['Coef.','[0.025', '0.975]']])
pct_change = (odds[['Coef.','[0.025', '0.975]']] - 1) * 100


corr_matrix(wrangle_features(DATA.drop(columns = ['action_type'])))
plot_proba(model2)
plot_regular_vs_post_season(model2)
plot_confusion_matrix(model2)



pdf_test_x = sm2.pdf(model2.test_x)

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



"""############# Model 4 - LDA - Predicted Log Loss: 9.351 #############"""

lda = LinearDiscriminantAnalysis()
lda = lda.fit(model2.train_x, model2.train_y)
lda_x = lda.transform(model2.train_x)
z = lda.transform(model2.test_x)
z_labels = lda.predict(model2.test_x)

log_loss(model2.test_y, z)


"""############# Model 5 #############"""

"""Dataset: d5 | Prediction set: d5_pred
    - shot_distance only predictor
"""

d5 = DATA[[DEPENDENT, 'shot_distance']]
# d3_pred = FOR_PREDICTION[d3cols].drop(columns = [DEPENDENT]).fillna(0)

"""Fit d5"""
d5 = sm.add_constant(d5)
model5 = LogR(d5, DEPENDENT)
model5.sm = model5.statsmodel()

model5.yhat = model5.sm.predict(model.test_x)

model5 = LogRegModel(d5)
s5 = model5.sm.summary2()

OR = np.exp(s5.tables[1]['Coef.'])


"""############# Model 6 - Log Loss: 0.669 #############"""

"""Dataset: d6 | Prediction set: d6_pred
    - Allen's model 5
"""

d6 = DATA[[DEPENDENT, 'shot_distance', 'playoffs', 'arena_temp', 'game_event_id', 'lat', 'lon']]

"""Fit d6"""
# d5 = sm.add_constant(d5)
# model5 = LogR(d5, DEPENDENT)


model6 = LogRegModel(d6)
s6 = model6.sm.summary2()


OR = np.exp(s6.tables[1]['Coef.'])

plot_confusion_matrix(model6)

model6.sensitivity()
model6.specificity()


#! Data Overview

# TODO: Add Univariate Plots
    # QQ
    # Hist

d6


d = d6 .select_dtypes(np.number)
