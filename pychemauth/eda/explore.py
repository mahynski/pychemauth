"""
Examine raw data used in ML and data science.

A collection of tools, from various sources, for data inspection and
exploratory data analysis (EDA) in ML and data science.  Attribution to
original sources is made available when appropriate.

author: nam
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm


class InspectData:
    """Class containing tools used to inspect raw data."""

    def __init__(self):
        """Initialize the class."""
        pass

    @staticmethod
    def cluster_elbow(X, clusters=range(1, 11)):
        """
        Compute cluster elbow metric.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.
        
        clusters : list(int), optional(default=range(1, 11))
            List of the number of clusters to use.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes the result is plotted on.

        Note
        ----
        Uses kmeans++ to examine within-cluster sum squared errors (inertia or
        distortion) and plots as the number of clusters increases.  Because of
        the use of kmeans, this is best if the clusters are more spherical.

        References
        ----------
        See Ch. 11 of "Python Machine Learning" by Raschka & Mirjalili.
        https://github.com/rasbt/python-machine-learning-book-2nd-edition

        See sklearn.cluster.KMeans.
        """
        from sklearn.cluster import KMeans

        distortions = []
        for i in clusters:
            km = KMeans(
                n_clusters=i,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=0,
            )
            km.fit(X)
            distortions.append(km.inertia_)
        plt.plot(clusters, distortions, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")
        plt.tight_layout()

        return plt.gca()

    @staticmethod
    def cluster_silhouette(X, clustering):
        """
        Plot silhouette curves.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.
        
        clustering : sklearn.cluster
            Clustering algorithm that implements a .fit_predict method, 
            e.g., sklearn.cluster.KMeans.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes the result is plotted on.

        Note
        ----
        Plot silhouette curves to assess the quality of clustering into a
        meaningful number of clusters in **classification tasks**. Ideal
        silhouette coefficients are close to 1, meaning "tight" well-separated
        clusters.

        References
        ----------
        See Ch. 11 of "Python Machine Learning" by Raschka & Mirjalili.
        https://github.com/rasbt/python-machine-learning-book-2nd-edition

        Example
        -------
        >>> km = KMeans(n_clusters=10,
                        init="k-means++",
                        n_init=10,
                        random_state=0)
        >>> cluster_silhouette(X, clustering=km)
        """
        from matplotlib import cm
        from sklearn import clone
        from sklearn.metrics import silhouette_samples

        # Clone clustering algorith, and predict clusters
        est = clone(clustering, safe=False)
        y = est.fit_predict(X)

        cluster_labels = np.unique(y)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, y, metric="euclidean")

        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / n_clusters)
            plt.barh(
                range(y_ax_lower, y_ax_upper),
                c_silhouette_vals,
                height=1.0,
                edgecolor="none",
                color=color,
            )

            yticks.append((y_ax_lower + y_ax_upper) / 2.0)
            y_ax_lower += len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--")

        plt.yticks(yticks, cluster_labels)
        plt.ylabel("Cluster")
        plt.xlabel("Silhouette coefficient")

        plt.tight_layout()

        return plt.gca()

    @staticmethod
    def cluster_collinear(
        X, feature_names=None, figsize=None, t=None, display=True, figname=None
    ):
        """
        Identify collinear features using the Spearman rank order correlation.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Dense feature matrix.

        feature_names : list, optional(default=None)
            Names of each column in X (in ascending order). If None, returns
            indices of columns, otherwise all outputs are in terms of names.

        figsize : tuple(int, int), optional(default=None)
            Size of visualization to display.  Ignored if display = False.

        t :  scalar(float), optional(default=None)
            Ward clustering threshold to determine the number of clusters.

        display : scalar(bool), optional(default=None)
            Whether or not to visualize results.

        figname : str, optional(default=None)
            If display is True, can also save to this file.

        Returns
        -------
        selected_features : ndarray(str or int, ndim=1)
            If feature names are provided, names are returned.  Otherwise they are
            the indices of the columns in X.

        cluster_id_to_feature_ids : defaultdict(list)
            Dictinary of {cluster id: features that belong}.

        fig : matplotlib.pyplot.figure or None
            Figure the result is plotted on if display = True, otherwise None.
            
        Note
        ----
        Ward clustering is used to cluster collinear features based on "distance",
        computed from Spearman rank order correlations, and select a single feature
        from each macro-cluster.

        Note that as of sklearn v1.0.2 the distance is linkage recommendation changed
        from directly using the correlation matrix to a distance metric.  Results
        appear largely similar qualitatively, but are not quantitatively identical.

        This can be used as a preprocessing step since it is unsupervised.

        References
        ----------
        See https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html

        Example
        -------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        >>> # Train model in first round
        >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> clf.score(X_test, y_test) # 97%
        >>> # Look at pfi --> EVERYTHING comes out as irrelevant because many features highly correlated
        >>> pychemauth.analysis.inspect.InspectModel.pfi(clf, X_test, y_test, n_repeats=30, 
        ... feature_names=data.feature_names.tolist())
        >>> # Look at multicollinearity
        >>> selected_features, cluster_id_to_feature_ids, _ =
        ... pychemauth.eda.explore.InspectData.cluster_collinear(X, # Unsupervised
        ...                                figsize=(12, 8),
        ...                                display=True,
        ...                                t=2,
        ...                                feature_names=None) # Get indices
        >>> # Fit again just using these selected features
        >>> X_train, X_test = X_train[:,selected_features],
        ... X_test[:,selected_features]
        >>> clf.fit(X_train, y_train)
        >>> clf.score(X_test, y_test) # 96%, almost identical as expected
        >>> # Top is "mean radius", which according to dendogram above, is
        ... highly correlated with other "size" metrics
        >>> pychemauth.analysis.inspect.InspectModel.pfi(clf, X_test, y_test, n_repeats=30,
        ... feature_names=data.feature_names[selected_features])
        """
        from collections import defaultdict

        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        from scipy.stats import spearmanr

        X = np.asarray(X, dtype=np.float64)
        if feature_names is None:

            def naming(i):
                return i

        else:
            feature_names = list(
                feature_names
            )  # Needs to be a list for compatibility elsewhere

            def naming(i):
                return feature_names[i]

        corr = spearmanr(X).correlation
        assert (
            np.max(np.abs(corr - (corr + corr.T) / 2.0)) < 1.0e-12
        ), "Spearman R matrix is not symmetric"
        corr = (corr + corr.T) / 2.0  # To make it perfectly symmetric
        np.fill_diagonal(corr, 1)

        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))

        # If no specification for where to cut, guess
        guess = np.sqrt(np.max(dist_linkage[:, 2])) if t is None else t

        cluster_ids = hierarchy.fcluster(
            dist_linkage, t=guess, criterion="distance"
        )
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(naming(idx))

        # Arbitrarily select the first feature put into each cluster
        selected_features = np.array(
            [v[0] for v in cluster_id_to_feature_ids.values()]
        )

        # Plot
        if display:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.axhline(guess, color="k")

            def decorate(x):
                return "***" + str(x).upper() + "***"

            if feature_names:
                labels = list(feature_names)
            else:
                labels = np.arange(X.shape[1]).tolist()
            for i in range(len(labels)):
                if labels[i] in selected_features:
                    labels[i] = decorate(labels[i])

            dendro = hierarchy.dendrogram(
                dist_linkage,
                ax=ax1,
                labels=labels,
                leaf_rotation=90,
                color_threshold=guess,
            )

            dendro_idx = np.arange(0, len(dendro["ivl"]))
            ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
            _ = ax2.set_xticks(dendro_idx)
            _ = ax2.set_yticks(dendro_idx)
            _ = ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
            _ = ax2.set_yticklabels(dendro["ivl"])
            _ = ax2.set_title("Spearman Rank-Order Correlations")

            if figname is not None:
                fig.tight_layout()
                plt.savefig(figname, dpi=300, bbox_inches="tight")
            else:
                fig.tight_layout()

        return (
            selected_features,
            cluster_id_to_feature_ids,
            fig if display else None,
        )

    def minimize_cluster_label_entropy(
        cluster_id_to_feature_ids,
        lookup,
        X,
        cutoff_factor=0.9,
        n_restarts=1,
        max_iters=1000,
        seed=0,
        early_stopping=-1,
        T=0.25,
    ):
        """
        Identify a set of features that are categorically similar.

        Parameters
        ----------
        cluster_id_to_feature_ids : defaultdict(list)
            Dictinary of {cluster id: features that belong}.

        lookup : callable
            A function that is capable of looking up a feature and returning a
            designation or class for it. For example lookup("mercury") =
            "Heavy Metal".  The final result is a set of features which all
            belong to the same class as much as possible.

        X : pandas.DataFrame
            DataFrame of the feature matrix.

        cutoff_factor : scalar(float), optional(default=0.9)
            Fraction of times a feature must be appear in X (not be NaN) to be
            considered as a viable feature.

        n_restarts : scalar(int), optional(default=1)
            Number of times to restart the Monte Carlo (annealing).

        max_iters : scalar(int), optional(default=1000)
            Maximum number of MC perturbations to make during a run.

        seed : scalar(int), optional(default=0)
            RNG seed to numpy.random.seed(seed).

        early_stopping : scalar(int), optional(default=-1)
            If the best (lowest entropy) solution hasn't changed in this many
            steps, stop the MC.

        T : scalar(float), optional(default=0.25)
            Fictitious "temperature" that controls acceptance criteria.

        Returns
        -------
        best_choices : list
            List of features to use.

        Note
        ----
        Minimize the entropy of selected features based on hierarchical
        clustering according to some labeling scheme that categorizes them.
        For example, lookup("mercury") = "Heavy Metal". This routine performs
        Metropolis Monte Carlo to minimize the entropy of the system defined by
        the categories of all features selected from each cluster.  Features
        are only considered viable if they appear in the input X DataFrame (not
        NaN values) at least 100*cutoff_factor percent of the time.

        This can be used to improve selections made at random by
        pychemauth.eda.explore.InspectData.cluster_collinear().

        Example
        -------
        >>> selected_features, cluster_id_to_feature_ids = 
        ... pychemauth.eda.explore.InspectData.cluster_collinear(X,
        ... feature_names=feature_names, display=False)
        >>> better_features =
        ... pychemauth.eda.explore.InspectData.minimize_cluster_label_entropy(
        ... cluster_id_to_feature_ids, lookup, X)
        """
        import copy

        np.random.seed(seed)

        # Python's default behavior is specify clusters starting from 1 not 0,
        # so change that.
        if np.all(
            sorted(cluster_id_to_feature_ids.keys())
            == np.arange(1, 1 + len(cluster_id_to_feature_ids))
        ):
            new_ids = copy.copy(cluster_id_to_feature_ids)
            for k, v in cluster_id_to_feature_ids.items():
                new_ids[k - 1] = v
            new_ids.pop(np.max(list(new_ids.keys())))
            cluster_id_to_feature_ids = new_ids
        elif np.all(
            sorted(cluster_id_to_feature_ids.keys())
            == np.arange(len(cluster_id_to_feature_ids))
        ):
            pass
        else:
            raise Exception("Cluster ID ordering not understood")

        if early_stopping <= 0:
            early_stopping = max_iters + 1

        # Determine which features are "safe" to use on the basis of a minimum
        # number of observations.
        safe_features = {}

        def counts(f):
            return X.shape[0] - X[f].isnull().sum()  # X is DataFrame

        cutoff = int(X.shape[0] * cutoff_factor)
        for cid, features in cluster_id_to_feature_ids.items():
            safe_features[cid] = [f for f in features if counts(f) > cutoff]
            assert (
                len(safe_features) > 0
            ), "Cutoff is too severe, no features allowed in \
                cluster {}".format(
                cid
            )

        # Look up all features to make sure lookup() works
        categories = set()
        for k, features_in in safe_features.items():
            try:
                categories = categories.union(
                    set([lookup(f) for f in features_in])
                )
            except Exception as e:
                raise ValueError(
                    "{} : Unable to lookup({}), check lookup \
                    function".format(
                        e, features_in
                    )
                )

        def random_choices(safe_features):
            choice_idx = {}
            for cid, features in safe_features.items():
                choice_idx[cid] = np.random.randint(len(features))
            return choice_idx

        def perturb(choice_idx, safe_features):
            new_choice_idx = copy.copy(choice_idx)
            cid = np.random.randint(len(new_choice_idx))
            new_choice_idx[cid] = np.random.randint(len(safe_features[cid]))

            return new_choice_idx

        def entropy(choices):
            cats, counts = np.unique(
                [lookup(v) for v in choices.values()], return_counts=True
            )
            p = counts / np.sum(counts, dtype=np.float64)
            return np.sum(-p * np.log(p))

        def convert(choice_idx, safe_features):
            choices = {}
            for cid, idx in choice_idx.items():
                choices[cid] = safe_features[cid][idx]
            return choices

        # Try to minimize the entropy of the cluster categories
        best_overall = (np.inf, None)
        for i in tqdm.tqdm(range(n_restarts), desc="Restarts"):
            # 1. Make a random selection of features
            choice_idx = random_choices(safe_features)
            S_curr = entropy(convert(choice_idx, safe_features))

            # 2. Perform MC to "anneal"
            best = (S_curr, choice_idx)
            counter = 0
            for j in tqdm.tqdm(range(max_iters), desc="MC Steps"):
                new_choice_idx = perturb(choice_idx, safe_features)
                S_new = entropy(convert(new_choice_idx, safe_features))
                if np.random.random() < np.exp(-(S_new - S_curr) / T):
                    # Accept move, always move down in entropy, but up
                    # stochastically.
                    choice_idx = new_choice_idx
                    S_curr = S_new
                    if S_curr < best[0]:
                        best = (S_curr, choice_idx)
                        counter = 0  # Reset time since last update.
                    else:
                        counter += 1
                else:
                    counter += 1

                if (
                    counter >= early_stopping
                ):  # Stop if we haven't found something better in a while.
                    break

            # 3. Compare across restarts
            if best[0] < best_overall[0]:
                best_overall = best

        # 4. Go over list and perform final optimization, choosing alternatives
        # of the same class that were the MOST often measured, not just AT
        # LEAST some minimum.
        # NOTE: THIS MIGHT CREATE SOME BIASED BASED ON LEXICOGRAPHIC ORDERING
        # I.E., IF TWO FEATURES HAVE THE SAME NUMBER OF OBSERVATIONS BUT DIFFERENT
        # NAMES THEN THE ONE WITH THE FIRST "NAME" WILL ALWAYS BE SELECTED.
        final = {}
        for cid, feat in convert(best_overall[1], safe_features).items():
            best_idx = sorted(
                [
                    (i, counts(f))
                    for i, f in enumerate(safe_features[cid])
                    if lookup(feat) == lookup(f)
                ],
                key=lambda x: x[1],
                reverse=True,
            )[0][0]
            final[cid] = safe_features[cid][best_idx]

        return list(final.values())

    @staticmethod
    def pairplot(df, figname=None, **kwargs):
        """
        Plot pairs of features against each other to look for trends.

        Parameters
        ----------
        df : DataFrame
            DataFrame with (dense) X predictors.  It may or may not contain a
            column for the prediction target.  For classification tasks, this
            can be visualized using "hue" as shown below.

        figname : str, optional(default=None)
            If not None the plot will be saved to this file.

        kwargs : dict
            Optional kwargs for seaborn.pairplot().

        Returns
        -------
        None

        Note
        ----
        A pairplot of the data.  Best to use after dimensionality reduction has
        been performed, e.g., using pychemauth.eda.explore.InspectData.cluster_collinear() 
        to select only certain features.  This can be helpful to visualize how 
        decorrelated the selected dimensions truly are.

        References
        ----------
        See https://seaborn.pydata.org/generated/seaborn.pairplot.html.

        Example
        -------
        >>> from sklearn.datasets import load_breast_cancer
        >>> data = load_breast_cancer()
        >>> df = pd.DataFrame(data=data.data, columns=data.feature_names)
        >>> df["target"] = data.target
        >>> InspectData.pairplot(df, vars=df.columns[0:5], hue="target",
        ... diag_kind="kde")
        >>> InspectData.pairplot(df, vars=df.columns[0:5], hue="target",
        ... diag_kind="auto")
        """
        sns.pairplot(df, **kwargs)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")

class JSBinary:
    """
    Look at pairwise "separability" according to the Jensen-Shannon divergence.

    Parameters
    ----------
    js_bins : scalar(int), optional(default=25)
        Number of bins to use when computing the Jensen-Shannon divergence.
        
    robust : scalar(bool), optional(default=False)
        Whether to robust option for JensenShannonDivergence.
            
    Note
    ----
    For a classification problem, look at the maximum JSD that can exists
    across all features between pairs of classes.  This creates a binary
    comparison between individual classes instead of a one-vs-all comparison 
    done in JSScreen.

    It can be helpful to look for the "elbow" as you plot number of bins vs.
    max JSD to get a sense for the optimal value.
    """

    def __init__(self, js_bins=25, robust=False):
        """Instantiate the class. """
        self.set_params(**{"js_bins": js_bins, "robust": robust})
        return

    def set_params(self, **parameters):
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {"js_bins": self.js_bins, "robust": self.robust}

    def fit(self, X, y):
        """
        Fit the screen to data.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Features matrix.
            
        y : array_like(str or int, ndim=1)
            Ground truth classes.
            
        Returns
        -------
        self : JSBinary
            Fitted model.
        """
        js = JensenShannonDivergence(
            **{
                "per_class": True,  # Sorts by max automatically
                "feature_names": None,  # Index
                "bins": self.js_bins,
                "robust": self.robust,
            }
        )

        self.__enc_ = LabelEncoder()
        self.__enc_.fit(y)
        self.__matrix_ = np.zeros(
            (len(self.__enc_.classes_), len(self.__enc_.classes_))
        )
        self.__top_feature_ = np.empty(
            (len(self.__enc_.classes_), len(self.__enc_.classes_)), dtype=object
        )
        for pairs in itertools.combinations(np.unique(y), r=2):
            # 2. Compute (max) JS divergence
            mask = (y == pairs[0]) | (y == pairs[1])

            # Binary so divergences are the same, just take the first
            div = js.fit(X[mask], y[mask]).divergence
            x = div[pairs[0]][0][1][pairs[0]]
            feature = div[pairs[0]][0][0]
            assert div[pairs[1]][0][1][pairs[1]] == x

            i, j = self.__enc_.transform(pairs)
            self.__matrix_[i][j] = x
            self.__matrix_[j][i] = x
            self.__top_feature_[i][j] = feature
            self.__top_feature_[j][i] = feature

        return self

    @property
    def matrix(self):
        """Return the matrix of maximum JS divergence values."""
        return self.__matrix_.copy()

    def top_features(self, feature_names=None):
        """
        Return which feature was responsible for the max JS divergence.

        Parameters
        ----------
        feature_names : array_like(str, ndim=1), optional(default=None)
            List of feature names. Results are internally stored as
            indices so if this is provided, converts indices to names
            based on this array; otherwise a matrix of indices is
            returned.
            
        Returns
        -------
        top_features : ndarray(object, ndim=2)
            Matrix of top feature names (if provided) or indices indicating
            the feature responsible for the maximum JSD between features i and
            j.  Diagonals are set to "NONE".
            
        Example
        -------
        >>> jsb.top_features(feature_names=X.columns)
        """
        if feature_names is None:
            return self.__top_feature_.copy()
        else:
            names = np.empty_like(self.__top_feature_)
            for i in range(names.shape[0]):
                for j in range(names.shape[1]):
                    if i != j:
                        names[i, j] = feature_names[self.__top_feature_[i, j]]
                    else:
                        names[i, j] = "NONE"
            return names

    def visualize(self, ax=None):
        """
        Visualize the results with a heatmap.
        
        Parameters
        ----------
        ax : matplotlib.pyplot.axes, optional(default=None)
            Axes to plot the result on.
            
        Returns
        -------
        ax : matplotlib.pyplot.axes
            Axes results are plotted on.
        """
        if ax is None:
            ax = plt.figure().gca()

        ax = sns.heatmap(
            self.matrix,
            ax=ax,
            annot=True,
            xticklabels=self.__enc_.classes_,
            yticklabels=self.__enc_.classes_,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(r"Maximum Pairwise $\nabla \cdot JS$")

        return ax