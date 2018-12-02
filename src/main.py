
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
		'game_date', # violates independence
		'action_type',
        'loc_x', # collinear with lat
        'loc_y', # collinear with lon

	]




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


"""

#* logOdds
-------------------------------------------------------------------------------------------
s                              Coef.    Std.Err.      z    P>|z|      [0.025       0.975]
lat                          -0.6565       0.3081 -2.1307 0.0331       -1.2604      -0.0526
lon                          -0.0526       0.1710 -0.3076 0.7584       -0.3879       0.2826
playoffs                      0.0417       0.0462  0.9019 0.3671       -0.0489       0.1324
seconds_remaining             0.0014       0.0009  1.5626 0.1182       -0.0004       0.0031
shot_distance                -0.0189       0.0044 -4.2505 0.0000       -0.0275      -0.0102
attendance                    0.0002       0.0000 10.3586 0.0000        0.0001       0.0002
arena_temp                    0.0328       0.0076  4.3008 0.0000        0.0178       0.0477
avgnoisedb                    0.0025       0.0078  0.3175 0.7509       -0.0128       0.0177
seconds_left_in_period        0.0001       0.0001  0.7850 0.4325       -0.0001       0.0002
last_seconds_of_period       -0.8629       0.1282 -6.7335 0.0000       -1.1141      -0.6117
seconds_left_in_game          0.0001       0.0000  2.2041 0.0275        0.0000       0.0001
home_or_away                 -0.0259       0.0305 -0.8465 0.3973       -0.0857       0.0340
num_shots_cumulative          0.0005       0.0042  0.1257 0.9000       -0.0077       0.0087
angle_from_basket             0.0002       0.0004  0.3873 0.6985       -0.0006       0.0009
season_count                  0.0093       0.0034  2.7567 0.0058        0.0027       0.0159

#* Odds

                              Coef.     Std.Err.       z  P>|z|  [0.025  0.975]

#!lat                          0.5175       0.3082 -2.1377 0.0325  0.2829  0.9467
lon                          0.9476       0.1711 -0.3144 0.7532  0.6777  1.3251
playoffs                     1.0214       0.0586  0.3602 0.7187  0.9105  1.1458
seconds_remaining            1.0014       0.0009  1.5652 0.1175  0.9996  1.0031
#!shot_distance                0.9813       0.0044 -4.2570 0.0000  0.9728  0.9899
game_date                    1.0002       0.0003  0.5708 0.5681  0.9995  1.0008
#!attendance                   1.0002       0.0000 10.3607 0.0000  1.0001  1.0002
#!arena_temp                   1.0331       0.0076  4.2622 0.0000  1.0177  1.0486
avgnoisedb                   1.0025       0.0078  0.3238 0.7461  0.9873  1.0180
seconds_left_in_period       1.0001       0.0001  0.7873 0.4311  0.9999  1.0002
#!last_seconds_of_period       0.4220       0.1282 -6.7317 0.0000  0.3283  0.5425
#!seconds_left_in_game         1.0001       0.0000  2.2072 0.0273  1.0000  1.0001
home_or_away                 0.9739       0.0306 -0.8659 0.3865  0.9173  1.0340
num_shots_cumulative         1.0005       0.0042  0.1242 0.9011  0.9923  1.0088
angle_from_basket            1.0002       0.0004  0.3880 0.6980  0.9994  1.0009
season_count                 0.9419       0.1212 -0.4942 0.6212  0.7427  1.1944

#* Percent Changes
                                   Coef.     Std.Err.       z  P>|z|    [0.025   0.975]
#!lat                             -48.1343       0.3081 -2.1307 0.0331  -71.6464  -5.1247
lon                              -5.1247       0.1710 -0.3076 0.7584  -32.1487  32.6623
playoffs                          4.2596       0.0462  0.9019 0.3671   -4.7756  14.1521
seconds_remaining                 0.1395       0.0009  1.5626 0.1182   -0.0354   0.3147
#!shot_distance                    -1.8678       0.0044 -4.2505 0.0000   -2.7173  -1.0109
#!attendance                        0.0173       0.0000 10.3586 0.0000    0.0141   0.0206
#!arena_temp                        3.3317       0.0076  4.3008 0.0000    1.7998   4.8866
avgnoisedb                        0.2476       0.0078  0.3175 0.7509   -1.2714   1.7900
seconds_left_in_period            0.0061       0.0001  0.7850 0.4325   -0.0092   0.0215
#!last_seconds_of_period          -57.8070       0.1282 -6.7335 0.0000  -67.1786 -45.7595
#!seconds_left_in_game              0.0070       0.0000  2.2041 0.0275    0.0008   0.0132
home_or_away                     -2.5519       0.0305 -0.8465 0.3973   -8.2134   3.4588
num_shots_cumulative              0.0527       0.0042  0.1257 0.9000   -0.7654   0.8775
angle_from_basket                 0.0150       0.0004  0.3873 0.6985   -0.0610   0.0911
season_count                      0.9308       0.0034  2.7567 0.0058    0.2681   1.5979

#* Significant features
                        Coef.  Std.Err.       z  P>|z|  [0.025  0.975]
lat                    0.5175    0.3082 -2.1377 0.0325  0.2829  0.9467
shot_distance          0.9813    0.0044 -4.2570 0.0000  0.9728  0.9899
attendance             1.0002    0.0000 10.3607 0.0000  1.0001  1.0002
arena_temp             1.0331    0.0076  4.2622 0.0000  1.0177  1.0486
last_seconds_of_period 0.4220    0.1282 -6.7317 0.0000  0.3283  0.5425
seconds_left_in_game   1.0001    0.0000  2.2072 0.0273  1.0000  1.0001

"""



""" #! Interpretation
The p value is calculated based on the assumption that the null hypothesis is true.

I think about it this way: “assuming the null hypothesis is true, the probability of the observed test statistic occurring is 0.02. That’s not very probable. But the observed test statistic definitely occurred, because it was observed. Therefore, it seems more likely that the null hypothesis is not true, i.e. It should be rejected.”

Assuming the null hypothesis is true, the probability of measuring at least the observed test occurring is 0.02.”

"""

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


# cNoFocus = "red"


#? Correlation Matrix


DATA.shape

#! better model was at the cost of explainability and violation of parsimony

"""
•	The __odds of Kobe making a shot decrease with respect to the distance he is from the hoop__.  If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

#! Yes. His odds go down by  -1.87% +-.85 %  (-2.72% -1.01%) for every additional foot away from the basket.
"""

"""
•	The __probability of Kobe making a shot decreases linearly with respect to the distance he is from the hoop__.    If there is evidence of this, quantify this relationship.  (CIs, plots, etc.)

#! It doesn't. Show pdf plot.

Linear up to 23ft, but is not at zero at 23ft, so probability curve must be curved.


"""



"""
•	The relationship between the __distance Kobe is from the basket and the odds of him making the shot is different if they are in the playoffs__.  Quantify your findings with statistical evidence one way or the other. (Tests, CIs, plots, etc.) 

#!

"""


""" Odds Ratios
Odds ratios that are greater than 1 indicate that the event is more likely to occur as the predictor increases. Odds ratios that are less than 1 indicate that the event is less likely to occur as the predictor increases.

https://www.predictiveanalyticsworld.com/patimes/on-variable-importance-in-logistic-regression/9649/

#! The model indicates a 4.25% increase in shooting ability during the playoffs, however, the result was not statistically significant.

CI's overlap and contain zero. zome evidence but not enough to conclude there is a difference.
