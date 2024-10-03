"""
Examine raw data used in ML and data science.

A collection of tools, from various sources, for data inspection and
exploratory data analysis (EDA) in ML and data science.  Attribution to
original sources is made available when appropriate.

author: nam
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from bokeh.io import output_notebook, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models import Slider # type: ignore[attr-defined]
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap
from sklearn.preprocessing import LabelEncoder

from typing import Any, Callable, Union, Sequence
from numpy.typing import NDArray

class InspectData:
    """Class containing tools used to inspect raw data."""

    def __init__(self) -> None:
        """Initialize the class."""
        pass

    @staticmethod
    def cluster_elbow(X: Union[NDArray[np.floating], Sequence[Sequence[float]]], clusters: Sequence[int] = range(1, 11)) -> matplotlib.pyplot.Axes:
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
        ax : matplotlib.pyplot.Axes
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
    def cluster_silhouette(X: Union[NDArray[np.floating], Sequence[Sequence[float]]], clustering: Any) -> matplotlib.pyplot.Axes:
        """
        Plot silhouette curves.

        Parameters
        ----------
        X : array_like(float, ndim=2)
            Feature matrix.

        clustering : sklearn.cluster
            Clustering algorithm that implements a `.fit_predict` method,
            e.g., sklearn.cluster.KMeans.

        Returns
        -------
        ax : matplotlib.pyplot.Axes
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
    def cluster_periodic_table(
        X: pd.DataFrame, step: float = 0.1, hover: bool = False, notebook_url: str = "http://localhost:8888"
    ) -> None:
        """
        Interactively cluster elements in the periodic table.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe of observations.  Columns should be elements, but may include other
            numeric measurements such as stable isotopes.  These will be ignored during
            clustering and coloring.

        step : scalar(float), optional(default=0.1)
            Increments to use on the slider.

        hover : bool, optional(default=False)
            Whether to show elemental properties when the mouse tip hovers over them.

        notebook_url : str, optional(default="http://localhost:8888")
            The URL of the notebook being used, including the port.  If you are running
            a Jupyter notebook server locally the default value is adequate; however, if
            you are accessing this remotely you need to specify the complete address so
            Bokeh can forward the plot correctly.  For example, http://123.4.567.890:8888.

        Note
        ----
        Clustering is performed using hierarchical Ward clustering as in
        `InspectData.cluster_collinear`.  The `t` value shown there is provided as an
        interactive slider.

        Warning
        -------
        Colors themselves are not meaningful; different colors represent different clusters
        but similar shades do NOT denote anything about the similarity of those clusters.
        """
        # Output should be set to notebook.
        output_notebook()

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame instance.")

        # Select elements from whatever is provided.
        known_elements = [str(e).lower() for e in elements.copy().symbol.values]
        used_elements = [
            str(c) for c in X.columns if str(c).lower() in known_elements
        ]
        X = X[used_elements]

        def create(doc):
            """Create the table."""
            periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
            groups = [str(x) for x in range(1, 19)]

            df = elements.copy()
            df["atomic mass"] = df["atomic mass"].astype(str)
            df["group"] = df["group"].astype(str)
            df["period"] = [periods[x - 1] for x in df.period]
            df = df[df.group != "-"]
            df = df[df.symbol != "Lr"]
            df = df[df.symbol != "Lu"]

            # For coloring
            df["cluster"] = ["0"] * df.shape[0]

            source = ColumnDataSource(df)

            TOOLTIPS = [
                ("Name", "@name"),
                ("Atomic number", "@{atomic number}"),
                ("Atomic mass (amu) ", "@{atomic mass}"),
                ("Type", "@metal"),
                ("CPK color", "$color[hex, swatch]:CPK"),
                ("Electronic configuration", "@{electronic configuration}"),
                ("Electronegativity", "@electronegativity"),
                ("Atomic Radius (pm)", "@{atomic radius}"),
                ("Ion Radius (pm)", "@{ion radius}"),
                ("VdW Radius (pm)", "@{van der Waals radius}"),
                ("Standard State", "@{standard state}"),
                ("Bonding Type", "@{bonding type}"),
                ("Melting Point (K)", "@{melting point}"),
                ("Boiling Point (K)", "@{boiling point}"),
                ("Density (g/m^3)", "@density"),
            ]

            p = figure(
                title="",
                width=1000,
                height=450,
                x_range=groups,
                y_range=list(reversed(periods)),
                tools="hover" if hover else "",
                toolbar_location=None,
                tooltips=TOOLTIPS if hover else None,
            )

            def recompute(attr, old, new):
                """Cluster and color elements."""
                (
                    selected_features,
                    cluster_id_to_feature_ids,
                    fig,
                ) = InspectData.cluster_collinear(
                    np.asarray(X.values, dtype=np.float64),
                    feature_names=X.columns,
                    display=False,
                    t=t_slider.value,
                )

                cm_ = matplotlib.colormaps["rainbow"].resampled(
                    len(cluster_id_to_feature_ids)
                )
                cmap = {"0": "#999d9a"}  # gray
                for idx, elements in sorted(
                    cluster_id_to_feature_ids.items(), key=lambda x: x[0]
                ):
                    cmap[str(idx)] = matplotlib.colors.rgb2hex(
                        cm_(idx - 1), keep_alpha=True
                    )
                    for elem in elements:
                        df["cluster"].where(
                            ~(
                                df["symbol"].apply(lambda x: str(x).lower())
                                == elem.lower()
                            ),
                            str(idx),
                            inplace=True,
                        )

                df.sort_values(
                    "cluster",
                    inplace=True,
                    key=lambda x: pd.Series([int(x_) for x_ in x]),
                )
                source.data = ColumnDataSource.from_df(df)

                # Unfortunately, there doesn't seem to be a way to link the color to the source.  Even
                # using a column in the df causes an error about waiting, so the best way forward seems
                # to be to re-build the table each time.
                r = p.rect(
                    "group",
                    "period",
                    0.95,
                    0.95,
                    source=source,
                    fill_alpha=1.0,
                    legend_field="cluster",
                    color=factor_cmap(
                        "cluster",
                        palette=list(cmap.values()),
                        factors=list(cmap.keys()),
                    ),
                )
                text_props = dict(
                    source=df,  # Leave unconnected from source since this doesn't need to be updated
                    text_align="left",
                    text_baseline="middle",
                    color="white",
                )
                x = dodge("group", -0.4, range=p.x_range)
                p.text(
                    x=x,
                    y="period",
                    text="symbol",
                    text_font_style="bold",
                    **text_props
                )
                p.text(
                    x=x,
                    y=dodge("period", 0.3, range=p.y_range),
                    text="atomic number",
                    text_font_size="11px",
                    **text_props
                )
                p.text(
                    x=x,
                    y=dodge("period", -0.35, range=p.y_range),
                    text="name",
                    text_font_size="7px",
                    **text_props
                )
                p.text(
                    x=x,
                    y=dodge("period", -0.2, range=p.y_range),
                    text="atomic mass",
                    text_font_size="7px",
                    **text_props
                )
                p.outline_line_color = None
                p.grid.grid_line_color = None
                p.axis.axis_line_color = None
                p.axis.major_tick_line_color = None
                p.axis.major_label_standoff = 0
                p.legend.orientation = "horizontal"
                p.legend.location = "top_center"
                p.hover.renderers = [r]

            # Build table as grid
            r = p.rect(
                "group",
                "period",
                0.95,
                0.95,
                source=source,
                fill_alpha=1.0,
                legend_field="cluster",
                color=factor_cmap(
                    "cluster", factors=["0"], palette=["#999d9a"]
                ),
            )

            # Build slider
            t_slider = Slider(
                start=0,
                end=2,
                value=0,  # Start visualization from t=0
                step=step,
                title="t value",
            )
            t_slider.on_change("value", recompute)

            # Color things for the first time
            recompute(None, None, None)

            doc.add_root(column(t_slider, p))

        show(create, notebook_url=notebook_url)

    @staticmethod
    def cluster_collinear(
        X: Union[NDArray[np.floating], Sequence[Sequence[float]]],
        feature_names: Union[Sequence[str], NDArray[np.str_], None] = None,
        figsize: Union[tuple[int, int], None] = None,
        t: Union[float, None] = None,
        display: bool = True,
        figname: Union[str, None] = None,
        highlight: bool = True,
        return_linkage: bool = False,
    ) -> list:
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

        display : bool, optional(default=None)
            Whether or not to visualize results.

        figname : str, optional(default=None)
            If display is True, can also save to this file.

        highlight : bool, optiona(default=True)
            If True, highlight the features selected on the output by adding
            asterisks and capitalization.

        return_linkage : bool, optional(default=False)
            If True, return the linkage matrix.

        Returns
        -------
        selected_features : ndarray(str or int, ndim=1)
            If feature names are provided, names are returned.  Otherwise they are
            the indices of the columns in X.

        cluster_id_to_feature_ids : defaultdict(list)
            Dictionary of {cluster id: features that belong}.

        fig : matplotlib.figure.Figure or None
            Figure the result is plotted on if `display` is True, otherwise None.

        dist_linkage : ndarray(float)
            If `return_linkage` is True then return the linkage used for hierarchical
            clustering.

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

        # This creates distances as if between fictitious points in a Euclidean
        # space, where the pairwise distances are defined by this correlation.
        # Subsequent agglomeration in heirarchy.ward assumes these are
        # Euclidean distances to perform clustering.
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
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
                if highlight:
                    return "***" + str(x).upper() + "***"
                else:
                    return str(x)

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
            ax1.set_ylabel("Distance")

            dendro_idx = np.arange(0, len(dendro["ivl"]))
            corr_ = ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
            _ = ax2.set_xticks(dendro_idx)
            _ = ax2.set_yticks(dendro_idx)
            _ = ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
            _ = ax2.set_yticklabels(dendro["ivl"])
            _ = ax2.set_title("Spearman Rank-Order Correlations")

            # Add space for color bar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
            fig.colorbar(corr_, cax=cbar_ax)

            if figname is not None:
                plt.savefig(figname, dpi=300, bbox_inches="tight")

        results = [
            selected_features,
            cluster_id_to_feature_ids,
            fig if display else None,
        ]
        if return_linkage:
            results += [dist_linkage]

        return results

    @staticmethod
    def minimize_cluster_label_entropy(
        cluster_id_to_feature_ids: dict[int, list],
        lookup: Callable[[Union[int, str]], Union[int, str]],
        X: pd.DataFrame,
        cutoff_factor: float = 0.9,
        n_restarts: int = 1,
        max_iters: int = 1000,
        seed: int = 0,
        early_stopping: int = -1,
        T: float = 0.25,
    ) -> list:
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
                len(safe_features[cid]) > 0
            ), "Cutoff is too severe, no features allowed in \
                cluster {}".format(
                cid
            )

        # Look up all features to make sure lookup() works
        categories: set = set()
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
            # <= ensures the first one is not kept by default
            if best[0] <= best_overall[0]:
                best_overall = best

        converted = convert(best_overall[1], safe_features)
        return [converted[k] for k in sorted(converted.keys())]

    @staticmethod
    def pairplot(df: pd.DataFrame, figname: Union[str, None] = None, **kwargs: Any) -> 'sns.PairGrid':
        """
        Plot pairs of features against each other to look for trends.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with (dense) X predictors.  It may or may not contain a
            column for the prediction target.  For classification tasks, this
            can be visualized using "hue" as shown below.

        figname : str, optional(default=None)
            If not None the plot will be saved to this file.

        kwargs : dict
            Optional kwargs for seaborn.pairplot().

        Returns
        -------
        pair : seaborn.PairGrid
            Seaborn PairGrid instance from seaborn.pairplot().

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
        pair = sns.pairplot(df, **kwargs)
        if figname is not None:
            plt.savefig(figname, dpi=300, bbox_inches="tight")

        return pair
