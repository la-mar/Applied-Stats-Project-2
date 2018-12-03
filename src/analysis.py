import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

from eda import *
from confusion_matrix_pretty import *
# from plotting import *

def pca_bin(train_x: pd.DataFrame, n_components = 2) -> pd.DataFrame:
    """Wrapper for sklearn.decomposition.PCA for binary classification

    Perform PCA analysis on X shaped (n_obs, n_features), returning X projected in the directions of PC1 and PC2.


    Arguments:
        train_x {pd.DataFrame} -- Matrix like object of independent variables

    Returns:
        pd.DataFrame -- DataFrame containing principle components
    """

    pca = PCA(n_components = n_components)
    result = pca.fit(train_x).transform(train_x)

    # Percentage of variance explained for each components
    print(f'''explained variance ratio:
        PC1: {round(pca.explained_variance_ratio_[0], 2)}
        PC2: {round(pca.explained_variance_ratio_[1], 2)} ''')

    result = pd.DataFrame(result,
                columns = ['PC1', 'PC2'],
                index = train_x.index)
    return result

def kaggle_test(cfr: str) -> pd.DataFrame:

    data = pd.read_csv('data/kaggle-data.csv')
    pred_x = data[data.shot_made_flag.isna()]
    pids = pred_x.shot_id

    data = wrangle_features(data)
    data = drop_features(data)
    data = fix_mulitcollinearity(data)

    pred_x = wrangle_features(pred_x)

    # Drop remaining categoricals
    data = data.drop(columns = [
                            'opponent',
                            'shot_type',
                            'combined_shot_type'
                        ]).dropna()

    pred_x = pred_x[data.columns].drop(columns =
                                    ['shot_made_flag'])

    model = cfr(data, DEPENDENT)
    model.fit()
    model.predict(pred_x)
    model.get_confusion_matrix()
    model_fi = model.feature_importance(thresh = None)

    pred_x['shot_made_flag'] = model.yhat
    pred_x['shot_id'] = pids
    pred_x[['shot_id', 'shot_made_flag']]
    pred_x.to_csv('data/to-kaggle.csv', index = False)


kaggle_test(LogR)