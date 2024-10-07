"""
Dealing with imbalanced datasets.

author: nam
"""
import numpy as np

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from sklearn.preprocessing import StandardScaler

from typing import Any, Union, ClassVar
from numpy.typing import NDArray


class ScaledSMOTEENN:
    """
    Balance a dataset.

    Parameters
    ----------
    sampling_strategy_smote : str, optional(default="not majority")
        Sampling strategy for SMOTE.

    sampling_strategy_enn : str, optional(default="not minority")
        Sampling strategy for ENN.

    k_smote : scalar(int), optional(default=5)
        Number of nearest neighbors to use for SMOTE over-sampling.

    k_enn : scalar(int), optional(default=3)
        Number of nearest neighbors to use for ENN under-sampling.

    kind_sel_enn : str, optional(default="all")
        ENN's strategy to exclude samples.

    random_state : scalar(int), optional(default=0)
        Sets the random state of the resamplers.

    scaler : sklearn.preprocessing object, optional(default=StandardScaler)
        Scaler, such as StandardScaler, to perform before SMOTE.

    Note
    ----
    See imblearn's documentation for details on these parameters.

    This uses the imblearn library to perform SMOTE-ENN, that is, SMOTE followed by edited nearest neighbors (ENN). This is done in a way that is compatible with imblearn's estimator API.  This class performs (1) Scaling, then (2) SMOTE-ENN, then (3) de-Scales to return a dataset in the original "units" provided.  The scaler can be chosen.

    Warning
    -------
    This is designed to work with **imblearn.pipeline** not sklearn.pipeline; they are often interchangeable, but not in this case.  Be sure to instantiate imblearn's version as in the example below.

    This is because imblearn's pipelines intermediates allow fit, transform, and resample methods; however, samplers are only applied during the fitting stage (training).

    References
    ---------
    https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTEENN.html#imblearn.combine.SMOTEENN

    Example
    -------
    >>> pipeline = imblearn.pipeline.Pipeline(steps=[
    ...     ("smote", ScaledSMOTEENN()),
    ...     ("scaler", StandardScaler()),
    ...     ("tree", DecisionTreeClassifier(random_state=0))
    ...     ])
    >>> param_grid = [
    ...     {'smote__k_enn':[3, 5],
    ...      'smote__kind_sel_enn':['all', 'mode'],
    ...      'tree__max_depth':[3,5]
    ...      }]
    >>> gs = GridSearchCV(estimator=pipeline,
    ...     param_grid=param_grid,
    ...     n_jobs=-1,
    ...     refit=True,
    ...     cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True))
    >>> gs.fit(X_imbalanced_train, y_imbalanced_train)
    """

    sampling_strategy_smote: ClassVar[str]
    sampling_strategy_enn: ClassVar[str]
    k_smote: ClassVar[int]
    k_enn: ClassVar[int]
    kind_sel_enn: ClassVar[str]
    random_state: ClassVar[int]
    scaler: ClassVar[Any]

    def __init__(
        self,
        sampling_strategy_smote: str = "not majority",
        sampling_strategy_enn: str = "not minority",
        k_smote: int = 5,
        k_enn: int = 3,
        kind_sel_enn: str = "all",
        random_state: int = 0,
        scaler: Any = StandardScaler(with_mean=True, with_std=True),
    ) -> None:
        """Instantiate the class."""
        self.set_params(
            **{
                "sampling_strategy_smote": sampling_strategy_smote,
                "sampling_strategy_enn": sampling_strategy_enn,
                "k_smote": k_smote,
                "k_enn": k_enn,
                "kind_sel_enn": kind_sel_enn,
                "random_state": random_state,
                "scaler": scaler,
            }
        )

    def set_params(self, **parameters: Any) -> "ScaledSMOTEENN":
        """Set parameters; for consistency with scikit-learn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "sampling_strategy_smote": self.sampling_strategy_smote,
            "sampling_strategy_enn": self.sampling_strategy_enn,
            "k_smote": self.k_smote,
            "k_enn": self.k_enn,
            "kind_sel_enn": self.kind_sel_enn,
            "random_state": self.random_state,
            "scaler": self.scaler,
        }

    def fit_resample(
        self,
        X: NDArray[np.floating],
        y: Union[NDArray[np.integer], NDArray[np.str_]],
    ) -> tuple[
        NDArray[np.floating], Union[NDArray[np.integer], NDArray[np.str_]]
    ]:
        """
        Resample the dataset.

        Parameters
        ----------
        X : ndarray(float, ndim=2)
            Feature matrix.

        y : ndarray(str or int, ndim=1)
            1-D array of classes.

        Returns
        -------
        X_resampled : ndarray(float, ndim=2)
            Resampled `X` and `y` in the original "units" of `X`.

        y_resampled : ndarray(str or int, ndim=1)
            Classes associated with each row in `X`.

        Note
        ----
        First standardize `X`, then perform SMOTEENN, then de-standardize to return results in the same "units" as input.
        """
        X_std = self.scaler.fit_transform(X)

        sm = SMOTEENN(
            smote=SMOTE(
                random_state=self.random_state,
                k_neighbors=self.k_smote,
                sampling_strategy=self.sampling_strategy_smote,
            ),
            enn=ENN(
                sampling_strategy=self.sampling_strategy_enn,
                n_neighbors=self.k_enn,
                kind_sel=self.kind_sel_enn,
            ),
        )

        X_res, y_res = sm.fit_resample(X_std, y)

        return self.scaler.inverse_transform(X_res), y_res
