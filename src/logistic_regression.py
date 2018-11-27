from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.metrics import (
        classification_report,
        roc_curve,
        auc

        )
import pandas as pd
from confusion_matrix_pretty import *
import statsmodels.api as sm


# m2b = LogR(d2, DEPENDENT)
# a = LogisticRegression(fit_intercept = True, C = 1e8)
# a.fit(m2b.train_x, m2b.train_y)
# a.score(m2b.test_x, m2b.test_y)

class LogR(LogisticRegression):
    """Sparse extension of sklearn.linear_model.LogisticRegression.

    """

    def __init__(self,
            data: pd.DataFrame,
            dependent_name: None,
            store_covariance: bool = True,
            test_size: float = 0.25,
            fit_intercept = False):
        super().__init__(solver = 'lbfgs',
                        fit_intercept = fit_intercept
                        )
        self.yhat = None

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data.drop(columns = [dependent_name]), data[dependent_name], test_size = test_size, random_state = 0)

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

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

    def statsmodel(self):
        """Model using statsmodels library.

        Returns:
            statsmodels result object

        """

        return sm.Logit(self.train_y, self.train_x).fit()

    def roc_plot(self, sm = True):
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



from sklearn.linear_model import LogisticRegression

def RecursiveFeatureSelection(X, y):
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(X.fillna(0), y.values.ravel())
    return rfe




# Evaluate Model
# https://www.r-bloggers.com/evaluating-logistic-regression-models/



#! http://blog.yhat.com/posts/logistic-regression-and-python.html

#! Dont use r-sq
# https://stats.stackexchange.com/questions/3559/which-pseudo-r2-measure-is-the-one-to-report-for-logistic-regression-cox-s

# TODO: Kobe last minute shots are outliers





# X_train, X_test, y_train, y_test = train_test_split(
#     d3.drop(columns=[DEPENDENT]),
#      d3[DEPENDENT], test_size=0.3, random_state=0)
# logreg = LogisticRegression(fit_intercept = True, C = 1e9)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# log_loss(y_test, y_pred)
# logreg.coef_

# # sm
# logit = sm.Logit(y_train, X_train)
# logit.fit().params









