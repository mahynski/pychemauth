"""
Inspect ML models.

A collection of tools, from various sources, for inspection of machine learning
models.  Attribution to original sources is made available when appropriate.

author: nam
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve


class InspectModel:
    """
    Inspect machine learning models.
    
    References
    ----------
    https://christophm.github.io/interpretable-ml-book/

    https://github.com/rasbt/python-machine-learning-book-2nd-edition
    """

    def __init__(self):
        """Initialize the class."""
        pass

    @staticmethod
    def confusion_matrix(model, X, y_true, ax=None):
        """
        Plot a confusion matrix for a classifier.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline
            Any function that implements a predict() method following
            sklearn's estimator API.

        X : array_like(float, ndim=2)
            Feature matrix for model to make predictions on.

        y_true : array_like(string or int, ndim=1)
            Correct labels.

        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot confusion matrix on.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes the confusion matrix has been plotted on.
            
        Notes
        -----
        Compare classification models based on true/false positive rates.
        
        References
        ----------
        See Ch. 6 of "Python Machine Learning" by Raschka & Mirjalili.
        https://github.com/rasbt/python-machine-learning-book-2nd-edition
        """
        confmat = confusion_matrix(y_true=y_true, y_pred=model.predict(X))

        if ax is None:
            fig = plt.figure()
            axes = plt.gca()
        else:
            axes = ax

        _ = sns.heatmap(
            confmat,
            ax=ax,
            annot=True,
            xticklabels=model.classes_,
            yticklabels=model.classes_,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        return ax

    @staticmethod
    def roc_curve(model, X, y, n_splits=10):
        """
        Select classification models based on true/false positive rates.
        
        Parameters
        ----------
        model : sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline
            Any function that implements a predict() method following
            sklearn's estimator API.

        X : array_like(float, ndim=2)
            Feature matrix for model to make predictions on.

        y : array_like(string or int, ndim=1)
            Class labels.

        n_splits : scalar(int), optional(default=10)
            N-fold CV to use.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes the ROC curve has been plotted on.
            
        References
        ----------
        See Ch. 6 of "Python Machine Learning" by Raschka & Mirjalili.
        https://github.com/rasbt/python-machine-learning-book-2nd-edition.
        """
        from scipy import interp
        from sklearn.metrics import auc, roc_curve

        _ = plt.figure()
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        cv = list(
            StratifiedKFold(n_splits=n_splits, random_state=0).split(X, y)
        )

        for i, (train, test) in enumerate(cv):
            probas = model.fit(X[train], y[train]).predict_proba(X[test])

            fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, label="ROC fold %d (area = %0.2f)" % (i + 1, roc_auc)
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color=(0.6, 0.6, 0.6),
            label="Random guessing",
        )

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(
            mean_fpr,
            mean_tpr,
            "k--",
            label="Mean ROC (area = %0.2f)" % mean_auc,
            lw=2,
        )
        plt.plot(
            [0, 0, 1],
            [0, 1, 1],
            linestyle=":",
            color="black",
            label="Perfect performance",
        )

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(loc="best")

        plt.tight_layout()

        return plt.gca()

    @staticmethod
    def learning_curve(
        model, X, y, cv=3, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0]
    ):
        """
        Diagnose bias/variance issues in a model.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Columns of features; observations are rows - will be converted to
            numpy array automatically.

        y : array_like(float, ndim=1)
            Response values.

        cv : scalar(int) or sklearn.model_selection object, optional(default=3)
            Cross-validation strategy; uses k-fold CV if an integer is provided.

        train_sizes : array_like(float, ndim=1)
            Fractions of provided data to use for training.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes the figure is plotted on.

        Notes
        -----
        The validation and training accuracy curves should converge "quickly"
        (if not, high variance) and to a "high" accuracy (if not, high bias).
        If it doesn't converge, it probably needs more data to train on.

        References
        ----------
        See Ch. 6 of "Python Machine Learning" by Raschka & Mirjalili.
        https://github.com/rasbt/python-machine-learning-book-2nd-edition

        Also see scikit-learn's documentation:
        https://scikit-learn.org/stable/modules/learning_curve.html

        Example
        -------
        >>> pipe_lr = make_pipeline(StandardScaler(),
        ... LogisticRegression(penalty='l2', random_state=1))
        >>> learning_curve(pipe_lr, X_train, y_train)
        """
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,  # Stratified by default in scikit-learn
            n_jobs=1,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.plot(
            train_sizes,
            train_mean,
            color="blue",
            marker="o",
            markersize=5,
            label="Training accuracy",
        )

        plt.fill_between(
            train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.15,
            color="blue",
        )

        plt.plot(
            train_sizes,
            test_mean,
            color="green",
            linestyle="--",
            marker="s",
            markersize=5,
            label="Validation accuracy",
        )

        plt.fill_between(
            train_sizes,
            test_mean + test_std,
            test_mean - test_std,
            alpha=0.15,
            color="green",
        )

        plt.grid()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.tight_layout()

        return plt.gca()

    @staticmethod
    def plot_residuals(y_true, y_pred):
        """
        Plot residuals and fit to a Gaussian distribution.

        Parameters
        ----------
        y_true : ndarray(float, ndim=1)
          N x K array of N observations made of K outputs.

        y_pred : ndarray(float, ndim=1)
          N x K array of N predictions of K variables. A model with a scalar
          output, for example, is just a column vector (K=1).

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes the figure is plotted on.
            
        Notes
        -----
        A good fit might indicate all predictive "information" has been
        extracted and the remaining uncertainty is due to random noise.
        """
        n_vars = y_true.shape[1]
        assert y_true.shape[1] == y_pred.shape[1]

        for i in range(n_vars):
            sns.jointplot(x=y_true[:, i], y=y_pred[:, i], kind="resid")

        return plt.gca()

    @staticmethod
    def pdp(model, X, features, **kwargs):
        """
        Partial dependence plots for features in X.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            A fitted scikit-learn estimator.

        X : array_like(float, ndim=2)
            Dense grid used to build the grid of values on which the dependence
            will be evaluated. **This is usually the training data.**

        features : list(int) or list(tuple(int, int))
            The target features for which to create the PDPs.
            If features[i] is an int, a one-way PDP is created; if
            features[i] is a tuple, a two-way PDP is created. Each tuple must
            be of size 2.

        Returns
        -------
        display : sklearn.inspection.PartialDependenceDisplay
            PDP display.
            
        Notes
        -----
        Partial dependence plots (PDP) show the dependence between the target
        response and a set of target features, marginalizing over the values of
        all other features (the complement features). Intuitively, we can
        interpret the partial dependence as the expected target response
        as a function of the target features.

        One-way PDPs tell us about the interaction between the target response
        and the target feature (e.g. linear, non-linear). Note that PDPs
        **assume that the target features are independent** from the complement
        features, and this assumption is often violated in practice.  If
        correlated features can be reduced, these might be more meaningful.

        PDPs with two target features show the interactions among the two
        features.

        References
        ----------
        See `sklearn.inspection.PartialDependenceDisplay`.

        Example
        -------
        >>> from sklearn.datasets import make_hastie_10_2
        >>> from sklearn.ensemble import GradientBoostingClassifier

        >>> X, y = make_hastie_10_2(random_state=0)
        >>> clf = GradientBoostingClassifier(n_estimators=100,
        ... learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
        >>> features = [0, 1]
        >>> InspectModel.pdp(clf, X, features)
        """
        from sklearn.inspection import PartialDependenceDisplay

        return PartialDependenceDisplay.from_estimator(model, X, features, **kwargs)

    @staticmethod
    def pfi(model, X, y, n_repeats=30, feature_names=None, visualize=False):
        """
        Compute permutation feature importances.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline
            A fitted estimator compatible with sklearn's estimator API.

        X : ndarray(float, ndim=2) or pandas.DataFrame
            Feature matrix model will predict.

        y : ndarray(float, ndim=1) or None
            Targets for supervised or None for unsupervised.

        n_repeats : scalar(int), optional(default=30)
            Number of times to permute a feature.

        feature_names : list(str), optional(default=None)
            Optional list of feature names in order.

        visualize : scalar(bool)
            Whether or not to visualize the results.

        Returns
        -------
        df : pandas.DataFrame
            Results in a dataframe.

        Notes
        -----
        Permutation feature importance is a model inspection technique that can
        be used for any fitted estimator **when the data is tabular.** The
        permutation feature importance is defined to be the decrease in a model
        score when a single feature value is randomly shuffled. It is
        indicative of **how much the model depends on the feature.**

        Can be computed on the training and/or test set (better).  There is
        some disagreement about which is actually better.  Scikit-learn says that:
        "Permutation importances can be computed either on the training set or
        on a held-out testing or validation set. Using a held-out set makes it
        possible to highlight which features contribute the most to the
        **generalization power** of the inspected model. Features that are
        important on the training set but not on the held-out set might cause
        the model to overfit."

        **Features that are deemed of low importance for a bad model (low
        cross-validation score) could be very important for a good model.**
        The pfi is only important if the model itself is good.

        The sums of the pfi should roughly add up to the model's accuracy (or
        whatever score metric is used), if the features are independent,
        however, unlike Shapley values, this will not be exact. In other
        words: results[results['95% CI > 0']]['Mean'].sum() /
        model.score(X_val, y_val) ~ 1.

        ``The importance measure automatically takes into account all
        interactions with other features. By permuting the feature you also
        destroy the interaction effects with other features. This means that
        the permutation feature importance takes into account both the main
        feature effect and the interaction effects on model performance. This
        is also a disadvantage because the importance of the interaction
        between two features is included in the importance measurements of both
        features. This means that the feature importances do not add up to the
        total drop in performance, but the sum is larger. Only if there is no
        interaction between the features, as in a linear model, the importances
        add up approximately.''
         - https://christophm.github.io/interpretable-ml-book/feature-importance.html

        For further advantages of pfi, see https://scikit-learn.org/stable/modules/permutation_importance.html. 
        One of particular note is that pfi place too much emphasis on 
        unrealistic inputs; this is because permuting features breaks 
        correlations between features.  If you can remove those correlations 
        then pfi's are more meaningful.

        When two features are correlated and one of the features is permuted,
        the model will still have access to the feature through its correlated
        feature. This will result in a lower importance value for both
        features, where they might actually be important.  One way to solve
        this is to cluster correlated features and take only 1.
        See :py:func:`eda.data.InspectData.cluster_collinear` for example.
        """
        import pandas as pd
        from sklearn.inspection import permutation_importance

        X = np.array(X)
        r = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=0
        )
        results = []

        def naming(i):
            return feature_names[i] if feature_names is not None else i

        for i in r.importances_mean.argsort()[::-1]:
            results.append(
                [
                    naming(i),
                    r.importances_mean[i],
                    r.importances_std[i],
                    r.importances_mean[i] - 2.0 * r.importances_std[i] > 0,
                ]
            )
        results = pd.DataFrame(
            data=results, columns=["Name or Index", "Mean", "Std", "95% CI > 0"]
        )

        if visualize:
            perm_sorted_idx = r.importances_mean.argsort()
            plt.boxplot(
                r.importances[perm_sorted_idx].T,
                vert=False,
                labels=feature_names[perm_sorted_idx],
            )

        return results
