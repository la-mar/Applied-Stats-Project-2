import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
from analysis import *
from confusion_matrix_pretty import *
from plotting import *

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

# Import Data
DATA = pd.read_excel('data/project2Data.xlsx', index = 'recId')
FOR_PREDICTION = pd.read_excel('data/project2Pred.xlsx')
DEPENDENT = "shot_made_flag"


"""Dataset: d1 | Prediction set: d1_pred
    - No categorical features
"""
d1 = prepare_data(DATA, drop_categorical = True) # Wrangle Data
d1 = d1.dropna()

d1_pred = prepare_data(FOR_PREDICTION)
d1_pred = d1_pred[cols(d1)].fillna(0)


"""Dataset: d2 | Prediction set: d2_pred
    - Categorical features as indicators
"""
d2 = prepare_data(DATA) 
d2 = get_dummies(d2) # Get dummy variables for categoricals

d2_pred = wrangle_features(FOR_PREDICTION)
d2_pred = get_dummies(d2_pred) # Get dummy variables for categoricals
d2_pred = d2_pred[cols(d2)].fillna(0)

pd.get_dummies(d2_pred, dtype = float).dropna().dtypes
d2.columns = d2.columns \
                    .str.lower() \
                    .str.replace(" ", "_")
"""Dataset: d3 | Prediction set: d3_pred
    - ???
"""
d3 = prepare_data(DATA, drop_categorical = True)
# d3 = DATA.drop(columns = ['opponent'])
d3 = get_dummies(d3)

d3_pred = wrangle_features(FOR_PREDICTION)
d3_pred = d3_pred[cols(d3)].dropna()


"""Fit d1"""

model = LogR(d1, DEPENDENT)
model.describe_features()
model.fit()
model.score()
model.predict(d1_pred)
model.log_loss()
model.confusion_matrix()
model.roc_plot()

model.sm = model.statsmodel()
model.sm.summary()
model.sm.summary2()
model.sm.wald_test_terms()

"""Fit d2"""

model = LogR(d2, DEPENDENT)
model.describe_features()
model.fit()
model.score()
model.predict(d2_pred)
model.log_loss()
model.confusion_matrix()
model.roc_plot()

model.sm = model.statsmodel()
model.sm.summary()
model.sm.summary2()
model.sm.wald_test_terms()


#! Data Overview

# TODO: Add Desc

# desc(test_x)

# TODO: Add Univariate Plots
    # QQ
    # Hist

"""
•	The __odds of Kobe making a shot decrease with respect to the distance he is from the hoop__.  If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

•	The __probability of Kobe making a shot decreases linearly with respect to the distance he is from the hoop__.    If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

•	The relationship between the __distance Kobe is from the basket and the odds of him making the shot is different if they are in the playoffs__.  Quantify your findings with statistical evidence one way or the other. (Tests, CIs, plots, etc.) 
"""


""" Odds Ratios
Odds ratios that are greater than 1 indicate that the event is more likely to occur as the predictor increases. Odds ratios that are less than 1 indicate that the event is less likely to occur as the predictor increases.

https://www.predictiveanalyticsworld.com/patimes/on-variable-importance-in-logistic-regression/9649/



"""
