import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
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



pd.options.display.max_rows = None
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('large_repr', 'truncate')
pd.set_option('precision',4)

# Matplotlib global config
plt.rcParams.update({'legend.fontsize': 'x-large',
		  'figure.figsize': (10, 6),
		 'axes.labelsize': 'small',
		 'axes.titlesize':'xx-large',
		 'xtick.labelsize':'small',
		 'ytick.labelsize':'small',
		 'savefig.dpi' : 300,
		 'savefig.format' : 'png',
		 'savefig.transparent' : True,
		 'axes.labelpad' : 10,
		 'axes.titlepad' : 10,
		 'axes.titleweight': 'bold'
		 })

plt.style.use('seaborn-deep')


# Define Contants

DEPENDENT = "shot_made_flag"
PERIODS_IN_GAME = 4
MIN_IN_PERIOD = 12
MIN_IN_GAME = MIN_IN_PERIOD * PERIODS_IN_GAME
SECONDS_IN_PERIOD = MIN_IN_PERIOD * 60
SECONDS_IN_GAME = MIN_IN_GAME * 60


"""
LOGISTIC MODEL:

- Dependent: shot_made: bool

"""

"""
LDA MODEL:

- Dependent: shot_made: bool

"""



"""
EDA:

- Potential Mulicolinearity:
	- Court Position: lat/log
					x/y
					shot_zone_area (cat)
					shot_zone_basic (cat)
					shot_zone_range (cat)

	- (maybe) game_date:


- Add Features:

	- game_count: cumulative number of games
		- "distance" between games is more or less equivalent (except between seasons), so representation as an ordinal continuous value is appropriate. The effects of season changes will still be captures by season_count.

	- home_or_away: home ("vs.") or away ("@")

	- seconds_left_in_game: apply function

	- seconds_left_in_period: min_remaining * 60 + seconds_remaining

	- season_count: cumulative number of seasons
		- "distance" between seasons is more or less equivalent, so representation as an ordinal continuous value is appropriate.

	- num_shots_cumulative: running total of number of shots up to the current point in the game

	- (NotYetImplemented) shot_difficulty

Stretch Features:

	- altitude: obtain from lat/long

	- central_angle_to_basket: instead of x/y

	- vector_length_to_basket: instead of x/y

Drop Features:

	- team_id: constant

	- team_name: constant

	- season: replace with season_count

	- game_id: replace with game_count

	- matchup: redundant with opponent

"""




def desc(df: pd.DataFrame):
	"""Produces a summary of the input DataFrame

	Arguments:
		df {pd.DataFrame} -- [description]

	Returns:
		pd.DataFrame -- DataFrame of summary statistics
	"""

	desc = df.describe().T
	desc['missing'] = len(df.index) - desc['count']
	# desc = desc.astype('int')
	desc['median'] = df.median()
	desc['missing %'] = desc.missing / len(df.index) * 100
	return desc.T

def vif(df: pd.DataFrame, dependent: str) -> pd.DataFrame:
	"""Get Variance Inflation Factor for each feature in df via a simple, multiple regression.

	Arguments:
		df {pd.DataFrame} -- dataset
		dependent {str} -- column name of dependent feature in df

	Returns:
		pd.DataFrame -- DataFrame containing feature names and VIF measures.
	"""

	# https://etav.github.io/python/vif_factor_python.html
	df = df.dropna()
	df = df._get_numeric_data() #drop non-numeric cols

	#gather features
	features = "+".join(df.columns.drop(dependent).tolist())

	# get y and X dataframes based on this regression:
	y, X = dmatrices('{} ~'.format(dependent) + features, df, return_type='dataframe')

	# For each X, calculate VIF and save in dataframe
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
	vif["features"] = X.columns

	return vif.round(1)

def angle(a: float, b: float, c: float) -> float:
	""" Calculate central angle for three known side lengths using Law of Cosines
	Arguments:
		a {Side} -- A side length
		b {Side} -- B side length
		c {Side} -- C side length

	Returns:
		Angle {float} -- central angle of A in degrees
	"""

	return degrees(acos((c**2 - b**2 - a**2)/(-2.0 * a * b)))

def central_angle(x: float, y:float) -> float:
	"""Calculate central angle of shot using NBA court grid coordinates.

	Arguments:
		x {float} -- X coordinate of shot
		y {float} -- Y coordinate of shot

	Returns:
		float -- angle in degrees of shot
	"""

	# Hack
	if (y == 0) & (x < 0):
		return -90

	if (y == 0) & (x > 0):
		return 90

	if (y == 0) & (x == 0):
		return 0

	# Vertices
	vc_a = (x, y) # shot loation
	vc_b = (0,0) # origin
	vc_c = (0, y) # reference point (0, y)

	side_a = distance.euclidean(vc_b, vc_c)
	side_b = distance.euclidean(vc_a, vc_c)
	side_c = distance.euclidean(vc_a, vc_b)

	# A = angle(side_a, side_b, side_c)
	# C = angle(side_c, side_a, side_b)
	B = angle(side_b, side_c, side_a)

	return B if x > 0 else -B

def wrangle_features(data: pd.DataFrame) -> pd.DataFrame:
	feats = pd.Series(
				data = False,
				index = ['recId',
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
						'avgnoisedb'],
				dtype = bool
				)

	# Flag features that were passed to the function
	feats.loc[feats.index.isin(data.columns)] = True
	try:
		if feats.minutes_remaining & feats.seconds_remaining:
			data['seconds_left_in_period'] = data.minutes_remaining * 60 + data.seconds_remaining
	except Exception as e:
		print('Failed to add feature: seconds_left_in_period. ({})'.format(e))
	try:
		if feats.period:
			data['seconds_elapsed_in_game'] = SECONDS_IN_PERIOD * data.period - data.seconds_left_in_period
	except:
		print('Failed to add feature: seconds_elapsed_in_game')

	try:
		if True:
			data['seconds_left_in_game'] = SECONDS_IN_GAME - data.seconds_elapsed_in_game
	except:
		print('Failed to add feature: seconds_left_in_game')

	try:
		if feats.matchup:
			data['home_or_away'] = data.matchup.str.contains("@").astype(int)
	except:
		print('Failed to add feature: home_or_away')

	try:
		if feats.game_id:
			data['num_shots_cumulative'] = data.groupby(['game_id']).cumcount()
	except:
		print('Failed to add feature: num_shots_cumulative')

	try:
		if feats.loc_x & feats.loc_y:
			data['angle_from_basket'] = data.apply(lambda row: central_angle(row.loc_x, row.loc_y), axis = 1)
	except:
		print('Failed to add feature: angle_from_basket')

	try:
		if feats.season:
			# Convert season to ordered Categorical (Factor) type
			data.season = pd.Categorical(data.season, data.season.sort_values().unique().tolist(), ordered = True)
			data['season_count'] = data.season.cat.codes
	except:
		print('Failed to add feature: season_count')

	try:
		if len(data.select_dtypes('object').columns):
			# Convert other string fields to unordered Categorical
			data[data.select_dtypes('object').columns.tolist()] = data.select_dtypes('object').astype('category')
	except:
		print('Failed to convert objects to categories')

	return data

def drop_redundant_features(data):

	# features to remove due to redundency or constant value
	remove = [
		'team_id',
		'team_name',
		'season',
		# 'game_id',
		'matchup',
		'shot_id',
		'recId',
		'shot_zone_area',
		'shot_zone_basic',
		'shot_zone_range',
		'minutes_remaining',
		'seconds_elapsed_in_game',
		# 'lat',
		# 'lon',
		# 'game_event_id',
		'game_date',
		'action_type'
	]

	# Remove columns in the 'remove' list if they are present in the dataset
	data = data.drop(columns = [x for x in remove if x in data.columns])
	return data

def eigen_solver():
	"""Assess using Eigen values and vectors

	https://stackoverflow.com/questions/25676145/capturing-high-multi-collinearity-in-statsmodels

	An almost zero eigen value shows a direction with zero variation, hence collinearity.

	"""
	# TODO: Implement, time permitting
	raise NotImplementedError()

def check_collinearity(data: pd.DataFrame):
	return vif(data, DEPENDENT) \
				.set_index('features') \
				.rename(columns = {'VIF Factor' : 'VIF'}) \
				.sort_values(by = 'VIF', ascending = False) \
				.drop('Intercept')

def check_collinearity_recursive(data: pd.DataFrame, vifs = None):
	"""Recursively check the multicollinearity (MC) associated with each feature.  Each iteration, the feature with the largest MC is dropped if the MC is infinite or if MC > x, where x is the standard deviation of the finite VIFs of the original features. A matrix containing VIFs for each iteration is returned once an iteration is reached where MC <= x.

	Arguments:
		data {pd.DataFrame} -- Matrix or DataFrame with shape(n_obs, n_features)

	Keyword Arguments:
		vifs {None} -- Recursive control parameter (default: {None})

	Returns:
		[pd.DataFrame] -- Matrix of VIFs per iteration. Nan (not a number) values represent features dropped from the assessment in either a previous or the current iteration.
	"""

	prev_vifs = vifs

	vifs = vif(data, DEPENDENT) \
				.set_index('features') \
				.rename(columns = {'VIF Factor' : 'VIF'}) \
				.sort_values(by = 'VIF', ascending = False) \
				.drop('Intercept') # Drop intercept term


	vif0_name, vif0_val = vifs.iloc[0].name, vifs.iloc[0].values[0]

	drop_feature = False
	limit = None
	thresh = prev_vifs.VIF[np.isfinite(prev_vifs.VIF)].max() if prev_vifs is not None else 0
	# If inflated feature VIF is infinite, drop the feature
	if vif0_val == float('inf'):
		drop_feature = True
	else:
		# Otherwise, drop feature if VIF within 2.5 stds.
		limit = (vifs[vifs != float('inf')].std()*2.5).values[0]
		if vif0_val > limit > thresh:
			drop_feature = True

	if prev_vifs is not None:
		# print('\n\nprev_vifs')
		# print(prev_vifs)
		vifs = prev_vifs.join(vifs, rsuffix = '_'+str(len(vifs)))



	print(f'VIF: Dropping: {vif0_name} | limit: {limit or 0:.2f} | thresh: {thresh or 0:.2f}')

	if drop_feature:
		return check_collinearity_recursive(
				data.drop(columns = [vif0_name]),
				vifs = vifs
				)

	print('\n\nvifs')
	print(vifs)

	return vifs

def fix_mulitcollinearity(data: pd.DataFrame):
	"""Remove multicollinear variables by assessing variance inflation factors.

	Arguments:
		data {pd.DataFrame} -- (n_obs, n_features)

	Returns:
		pd.DataFrame -- data
	"""
	print('\n\n')
	vifs = check_collinearity_recursive(data)
	# vifs = check_collinearity(data)
	vifs = vifs.iloc[:, -1] # Get last column (the last iteration)

	# remove features with high MC from data set
	data = data.drop(columns = vifs[vifs.isna()].index)
	return data

def prepare_data(data: pd.DataFrame, drop_categorical = False) -> pd.DataFrame:
	"""Template procedue to ingest new dataset.

	Arguments:
		data {pd.DataFrame} -- new dataset

	Returns:
		pd.DataFrame -- dataset for further prep or analysis
	"""

	data = wrangle_features(data)
	data = drop_redundant_features(data)
	data = fix_mulitcollinearity(data)

	# Drop remaining categoricals
	if drop_categorical:
		data = data.select_dtypes(exclude = ['object', 'category'])
		# data.drop(
		# 				columns = [
		# 					'opponent',
		# 					'shot_type',
		# 					'combined_shot_type',
		# 					'action_type'
		# 				]
		# 				)

	# print('\n\ndtypes:')
	# print(d1.dtypes)
	return data




# data = data.drop(columns = [
#     'minutes_remaining',
#     'seconds_remaining',
#     'seconds_elapsed_in_game',
#     'lat',
#     'lon',
#     'game_event_id',
#     'period',
#     'seconds_left_in_period',]
#     )

# Sort features by VIF

# NOTE: PLOTS

# fig, ax = plt.subplots(figsize=(12,8))
# ax = sns.scatterplot('loc_x', 'loc_y', hue = 'shot_made_flag', data = data)
# ax.set_title('Shot Location')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_ylim(0, 400)
# fig.savefig('figs/p2-3_price-v-months.png')

# fig, ax = plt.subplots(figsize=(12,8))
# ax = data.boxplot()
# ax.set_xticklabels(data.columns, rotation=90)
# fig.tight_layout()

# fig, ax = plt.subplots(figsize=(12,8))
# ax = data.select_dtypes(include=[np.number]).hist()
# # ax.set_xticklabels(data.columns, rotation=90)
# fig.tight_layout()

# import time
# from sklearn.linear_model import LassoCV
# print("Computing regularization path using the coordinate descent lasso...")
# t1 = time.time()
# model = LassoCV(cv=5).fit(X, y)
# t_lasso_cv = time.time() - t1

# # Display results
# m_log_alphas = -np.log10(model.alphas_)

# plt.figure()
# ymin, ymax = 2300, 3800
# plt.plot(m_log_alphas, model.mse_path_, ':')
# plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
#          label='Average across the folds', linewidth=2)
# plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
#             label='alpha: CV estimate')

# plt.legend()

# plt.xlabel('-log(alpha)')
# plt.ylabel('Mean square error')
# plt.title('Mean square error on each fold: coordinate descent '
#           '(train time: %.2fs)' % t_lasso_cv)
# plt.axis('tight')
# plt.ylim(ymin, ymax)


# def correct_multicollinearity(data: pd.DataFrame) -> pd.DataFrame:
# 	print('Anterior VIF')
# 	print(vif(data, DEPENDENT))

# 	# Drop multicolinear features
# 	data = data.drop(columns = [
# 		'minutes_remaining',
# 		'seconds_remaining',
# 		'seconds_elapsed_in_game',
# 		'lat',
# 		'lon',
# 		'game_event_id',
# 		'period',
# 		'seconds_left_in_period',

# 	])

# 	print('Posterior VIF')
# 	v = vif(data, DEPENDENT)
# 	print(v)
# 	return data