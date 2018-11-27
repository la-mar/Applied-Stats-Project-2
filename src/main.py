
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
from plotting import *
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
    model = LogR(data.fillna(0), DEPENDENT)
    # print(model.describe_features())
    model.sm = model.statsmodel()
    # print(model.sm.summary())
    # print(model.sm.summary2())
    model.sm.wald_test_terms()
    model.yhat = model.sm.predict(model.test_x)
    print("\nLog Loss: {}\n".format(
        round(
            log_loss(model.test_y, model.yhat)
            , 4)))
    return model


# Import Data
DATA = pd.read_excel('data/project2Data.xlsx', index = 'recId')
FOR_PREDICTION = pd.read_excel('data/project2Pred.xlsx')
DEPENDENT = "shot_made_flag"






"""########################### Model 1 ###########################"""

"""Dataset: d1 | Prediction set: d1_pred
    - No categorical features
"""
d1 = prepare_data(DATA, drop_categorical = True) # Wrangle Data
d1 = d1.dropna()
d1_pred = wrangle_features(FOR_PREDICTION)
d1_pred = d1_pred[cols(d1)].fillna(0)

"""Fit d1"""
model1 = LogRegModel(d1)
model1.describe_features()
model1.sm.summary2()

"""########################### Model 2 ###########################"""

"""Dataset: d2 | Prediction set: d2_pred
    - Categorical features as indicators
"""
d2 = prepare_data(DATA)
d2 = get_dummies(d2).fillna(0) # Get dummy variables for categoricals
d2_pred = wrangle_features(FOR_PREDICTION)
d2_pred = get_dummies(d2_pred) # Get dummy variables for categoricals
d2_pred = d2_pred[cols(d2)].fillna(0)

"""Fit d2"""

model2 = LogRegModel(d2)
model2.describe_features()
model2.sm.summary2()
model2.roc_plot()

"""########################### Model 3 ###########################"""

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
model3.describe_features()
model3.sm.summary2()

"""########################### Model 4 ###########################"""

"""Dataset: d4 | Prediction set: d4_pred
    - Categorical features as indicators
    - log shot_distance
    - log time_remaining
"""


# d4 = DATA[DATA.seconds_left_in_period > 1]
d4 = prepare_data(d4)
# d4.shot_distance = np.log(d4.shot_distance)
d4 = get_dummies(d4).fillna(0) # Get dummy variables for categoricals
# d4.game_date = d4.game_date.apply(lambda x: x.toordinal())
d4 = d4.drop(columns = ['game_date'])
d4_pred = wrangle_features(FOR_PREDICTION)
d4_pred = get_dummies(d4_pred) # Get dummy variables for categoricals
d4_pred = d4_pred[cols(d4)].fillna(0)



"""Fit d4"""
# model4 = LogRegModel(d4)

model4 = LogR(d4, DEPENDENT)
model4.describe_features()
model4.sm.summary2()

print(model.describe_features())
model4.sm = sm.Logit(model.train_y, sm.add_constant(model4.train_x)).fit()
model4.sm.summary2()
model4.sm.wald_test_terms()
model4.yhat = model4.sm.predict(model4.test_x)
print("\nLog Loss: {}\n".format(
    round(
        log_loss(model4.test_y, model4.yhat)
        , 4)))

#! Data Overview

# TODO: Add Univariate Plots
    # QQ
    # Hist

sm.qqplot(DATA.arena_temp, stats.t, fit=True, line='45')

#? Correlation Matrix
import statsmodels.graphics.api as smg
d = DATA.select_dtypes(np.number)
corr_matrix = np.corrcoef(d.T)
smg.plot_corr(corr_matrix, xnames=d.columns)


clf = LogisticRegression()
clf.fit(model2.train_x, model2.train_y)
preds = clf.predict_proba(model2.test_x)[:,1]
fpr, tpr, _ = roc_curve(model2.test_y, preds)

# fpr, tpr, _ = roc_curve(self.test_y, y_score)

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=1,
            label='ROC fold %d (AUC = %0.2f)' % (1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

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


