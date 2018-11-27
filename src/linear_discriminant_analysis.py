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
from confusion_matrix_pretty import *
from plotting import *

class LDAB(LinearDiscriminantAnalysis):
    """Sparse extension of sklearn.discriminant_analysis.LinearDiscriminantAnalysis for handling binary class cases.

    """

    def __init__(self,
            data: pd.DataFrame,
            dependent_name: None,
            store_covariance: bool = True,
            test_size: float = 0.25):
        super().__init__(
                        n_components = 2,
                        store_covariance = store_covariance
                        )
        self.yhat = None

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data.drop(columns = [dependent_name]), data[dependent_name], test_size = test_size, random_state = 0)

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def explained_variance(self) -> float:
        """Get variance explained per discriminant.

        Returns:
            float -- explained variance ratio
        """

        print(f'''Explained variance ratio:
        Discriminant 1: {self.explained_variance_ratio_[0]: .2f}''')
        # return self.explained_variance_ratio_[0]

    def describe_features(self):

        print(f"""
        X: features: {len(self.train_x)}

            dtypes:
            -------""")

        for k, v in self.train_x.dtypes.sort_index().items():
            print(f'''\t\t{k:<30}{v.name:} ''')

        print(f"""
        Y: {self.train_y.name}: {self.train_y.dtype}
        """)

    def confusion_matrix(self, test = False, plot = True):
        """Generate, and optionally plot, a confusion matrix for the test or train datasets

        Keyword Arguments:
            plot {bool} -- optionally plot the confusion matrix (default: {True})

        Returns:
            pd.DataFrame -- confusion matrix
        """
        if test:
            x = self.test_x
            y = self.test_y
        else:
            x = self.train_x
            y = self.train_y

        # if y is not None and x is not None:
        self.cm = pd.DataFrame(confusion_matrix(y, self.predict(x)), index = [0, 1], columns = [0, 1])
        if plot:
            pretty_plot_confusion_matrix(self.cm, cmap='PuRd')
        return self.cm
        # else:
        #     print('X or Y is empty. Check parameters.')
        #     return None

    def score(self) -> float:
        """Wrapper for parent class method using xy's stored in child class object.  Scores model fit using test data.

        Returns:
            float -- model score

        """

        x = self.test_x
        y = self.test_y

        score = super().score(x, y)
        print(f'''
        Features:
            {' | '.join([x for x in x.columns])}

        Accuracy: {score:.2%}

        ''')
        return score

    def plot_separability() -> None:
        """Plots a heatmap of fitted coefficients, highlighting features that are more likely seperable by a linear hyperplane.
        """

        x = self.train_x
        if len(x.columns.tolist()) == len(self.coef_[0]):
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sns.heatmap(pd.DataFrame(self.coef_[0],
                                    columns=[1],
                                    index=x.columns.tolist()),
                        ax=ax, cmap='RdBu', annot=True)

            plt.title('LDA Feature Separability')
            plt.tight_layout()
        else:
            print('Length of input "x" does not match number of coefficients. Refit the model using the dependents in x.')
            return None

    def fit(self):
        """Wrapper for parent class method using xy's stored in child class object.

        Fit the model.

        Returns:
            pd.Series -- array of x values projected to maximize seperation

        """
        super().fit(self.train_x, self.train_y)

    def predict(self, x):

        self.yhat = super().predict(x)
        return self.yhat

    def transform(self):
        """Wrapper for parent class method using xy's stored in child class object.

        Transform x to maximize seperation.

        Returns:
            pd.Series -- array of x values projected to maximize seperation

        """
        return super().transform(self.x)

    def log_loss(self, x = None):

        if self.yhat is not None:
            self.yhat = self.predict(x or self.test_x)

        return round(log_loss(self.test_y, self.yhat), 2)

    def _decision_function(self):
        #TODO: Implement, time permitting
        raise NotImplementedError()

    def classification_report(self):
        """Class wrapper for sklearn.metrics.classification_report
        """

        classification_report(
            self.test_y,
            self.yhat,
            target_names=self.classes_.astype(str).tolist())

    def roc_plot(self):
        """
        Referenced from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        """

        y_score = self.decision_function(self.test_x)

        fpr, tpr, _ = roc_curve(self.test_y, y_score)

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
Assumptions:
    - Each class must be:
        - normally distributed
        - identical cov matrices
        - independent

"""

if __name__ == "__main__":


    #! Try full model
    model = LDAB(data.fillna(0), DEPENDENT)
    model.fit()
    model.explained_variance()
    model.score()
    model.get_confusion_matrix()
    model.plot_separability()
    model_fi = model.feature_importance()
    model.log_loss()

    #! Try reduced model
    data_reduced = data[model_fi.index]

    ldab_reduced = LDAB(data_reduced, DEPENDENT)
    ldab_reduced.fit()
    ldab_reduced.explained_variance()
    ldab_reduced.score_()
    ldab_reduced.get_confusion_matrix()
    ldab_reduced.plot_separability()
    ldab_reduced.log_loss()

    disc1 = ldab_reduced.fit_transform(train_x_reduced, train_y)


    # Plot single discriminant
    sns.distplot(disc1)

    # TODO: Add plots

    #! Yes! Shows no seperation

    sns.pairplot(temp_x,
            hue="shot_made_flag",
            palette="husl",
            markers = ['<', '>'],
            plot_kws = {
                'alpha': 0.5,
            },
            diag_kws = {

            },
            )

    """
    Conclusion:

    Reduced model, with only 2 features, yields results effectively equivalent to those of the full model that uses 13 features.
    """

