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
from sklearn.metrics import confusion_matrix

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
        self.cm = pd.DataFrame(confusion_matrix(y, self.sm2.fitted.predict(x)), index = [0, 1], columns = [0, 1])
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
            self.yhat = self.sm.predict(x or self.test_x)

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

    def statsmodel_(self):
        """Model using statsmodels library.

        Returns:
            statsmodels result object

        """

        return sm.Logit(self.train_y, self.train_x)


    def roc_plot(self, sm = True):
        """
        Referenced from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
        """

        y_score = self.predict_labels(self.test_x)

        fpr, tpr, _ = roc_curve(self.test_y, y_score)

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=1,
                    label='ROC fold %d (AUC = %0.2f)' % (1, roc_auc))

        g = plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        g.set_title(f'Distribution of {y}', color = cNoFocus)
        g.set_xlabel(f'{y}', size = 'xx-large', color = cNoFocus)
        g.set_ylabel(f'Density', size = 'xx-large', color = cNoFocus)
        g.set_xticklabels(g.get_xticklabels(), size = 'xx-large')
        g.set_yticklabels(g.get_yticklabels(), size = 'xx-large')
        g.tick_params(colors=cNoFocus)
        g.spines['bottom'].set_color(cNoFocus)
        g.spines['top'].set_color(cNoFocus)
        g.spines['left'].set_color(cNoFocus)
        g.spines['right'].set_color(cNoFocus)
        g.xaxis.label.set_color(cNoFocus)
        g.yaxis.label.set_color(cNoFocus)
        return g


    def predict_labels(self, x, thresh = 0.5):
        """Predict class labels"""

        self.yhat = self.sm.fitted.predict(x)
        pc = np.zeros(len(self.yhat))
        pc[self.yhat > thresh] = 1
        return pc

    def sensitivity(self):
        """Sensitivity - TP/(TP+FN)"""

        cm = self.sm.pred_table()
        sens = cm[0,0]/(cm[0,0]+cm[0,1])
        print('Sensitivity : ', sens )
        return sens


    def specificity(self):
        """Specificity = TN/(TN+FP)"""

        cm = self.sm.pred_table()
        spec = cm[1,1]/(cm[1,0]+cm[1,1])
        print('Specificity : ', spec)
        return spec




def RecursiveFeatureSelection(X, y):
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(X.fillna(0), y.values.ravel())
    return rfe


