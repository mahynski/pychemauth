Python-based Chemometric Authentication
==========================
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![Workflow](https://github.com/mahynski/chemometrics/actions/workflows/python-app.yml/badge.svg?branch=main)
<!--[![codecov](https://codecov.io/gh/mahynski/chemometrics/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/chemometrics)-->

This is a centralized repository of tools implemented in python to perform chemometric authentication; the "other" PCA!  These methods are designed to be compatible with [scikit-learn's estimator API](https://scikit-learn.org/stable/developers/develop.html) so they can be deployed in pipelines used with GridSearchCV, etc.  Authentication is typically a [one-class classification (OCC)](https://en.wikipedia.org/wiki/One-class_classification), or class modeling, approach designed to detect anomalies. This contrasts with multi-class classification (discrimination) models which involve supervised learning of multiple classes to distinguish between them; the primary weakness of this is that such a model typically cannot predict if a new sample belongs to **none** of the classes trained on.

> "Outlier detection and novelty detection are both used for anomaly detection, where one is interested in detecting abnormal or unusual observations. Outlier detection is then also known as unsupervised anomaly detection and novelty detection as semi-supervised anomaly detection. In the context of outlier detection, the outliers/anomalies cannot form a dense cluster as available estimators assume that the outliers/anomalies are located in low density regions. On the contrary, in the context of novelty detection, novelties/anomalies can form a dense cluster as long as they are in a low density region of the training data, considered as normal in this context." - [scikit-learn's documentation](https://scikit-learn.org/stable/modules/outlier_detection.html)

Essentially, outlier detection methods characterize inliers as those points in high density regions, whereas novelty detection routines try to characterize a boundary around the region where a known class is found (even if it disperse). Both can be useful when attempting to detect chemometric anomalies.

## License Information
* See LICENSE for more information.
* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by [NIST](https://www.nist.gov/).

# Capabilities

![](pychemauth.png)

## Preprocessors
[scikit-learn](https://scikit-learn.org) provides a number of other simple [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) steps, including data standardization and imputation approaches.  Here, these are extended to include:

### Imputing Missing Data
* [Expectation Maximization with Iterative PCA (missing X values)](https://www.sciencedirect.com/science/article/pii/S0169743901001319?casa_token=PJMbl_1gHmoAAAAA:0Q4M969UyZ-MYQY44S0dFMtH77aX-AOxcCRSFBaDHuvsd2UnulLO3cUxh5GlHXnyJBzSp3oneO00)
* [Expectation Maximization with Iterative PLS (missing X values)](https://www.sciencedirect.com/science/article/pii/S0169743901001319?casa_token=PJMbl_1gHmoAAAAA:0Q4M969UyZ-MYQY44S0dFMtH77aX-AOxcCRSFBaDHuvsd2UnulLO3cUxh5GlHXnyJBzSp3oneO0)
* Limit of Detection (randomly below LOD)

### Scaling
* Corrected Scaling (akin to scikit-learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) but uses [unbiased/corrected standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation) instead)
* Pareto Scaling (scales by square root of standard deviation)
* Robust Scaling (scales by IQR instead of standard deviation)

### Filtering
* [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)

### Generating Synthetic Data
Can be used to balance classes during training, or to supplement measurements that are very hard to make.  `imblearn` pipelines are designed to be drop-in replacements for sklearn pipelines so are often used instead.
* [Imbalanced Learning](https://imbalanced-learn.org/stable/index.html) - SMOTE, ADASYN, etc.
* Generative Networks (VAE, GAN)

## Conventional Chemometrics [Small amount of data available]

> [Conventional chemometric authentication methods](https://www.sciencedirect.com/science/article/pii/S0003267017306050?casa_token=7wJt53xzxFgAAAAA:LSvTEjSKSTXsoFfH71ccxnP5eOj9OX3VxnPhA1t02FYYfsKosJQjq3s-rgKJUX0VNu7sFrrYvbA) generally fall under the umbrella of multivariate regression or classification tasks.  For example, the model proposed when performing multilinear regression is  `y = MX + b`, where the matrix `M` must be solved for. (Un)supervised classification is commonly performed via [projection methods](https://www.sciencedirect.com/science/article/pii/S0169743902001077?casa_token=Drui6g1wMgQAAAAA:qG1E9HHTSWrM1UhkWnLWw2iBxFAOa0Qsi9LblalX4PvfLCHNay0m-besnzOyZwXtBfI4LLGp7wQ), which compress the data into a lower dimensional space. A common choice of data models is: `X = TP^T + E`, where the scores matrix, `T`, represents the projection of the `X` matrix into a (lower dimensional) score space. The `P` matrix, called the [loadings matrix](http://www.statistics4u.com/fundstat_eng/cc_pca_loadscore.html), is computed in different ways.  For example, PCA (unsupervised) uses the leading eigenvectors of the covariance matrix of `X`, where as PLS uses a different (supervised) decomposition which is a function of both `X` and `y`. `E` is the error resulting from this model.

> OCC methods require careful preparation of the training set to remove extremes and outliers so that "masking" effects do not affect your final model. Manual data inspection is typically (but not always) required. Thus, conventional authentication methods can be considered [novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html) methods (no outliers in training), but many have built in capabilities to interatively "clean" the training set if outliers are assumed to be present initially. See ["Detection of Outliers in Projection-Based Modeling" by Rodionova and Pomerantsev](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b04611) for an example of outlier detection and removal in projection-based modeling.

### Classifiers
* PCA (for data inspection)
* PLS-DA (soft and hard variants) - [discriminant analysis is not the same as OCC for authentication](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3030).
* SIMCA
* DD-SIMCA
<!-- UNEQ, PFM see P. Oliveri / Analytica Chimica Acta 982 (2017) 9-19 -->

### Regressors
* PCR
* PLS

### Non-parametric Methods
* SRD (Sum of Ranking Differences)

### Some recent references:
* [Morais, Camilo LM, et al. "Tutorial: multivariate classification for vibrational spectroscopy in biological samples." Nature Protocols 15.7 (2020): 2143-2162.](https://www.nature.com/articles/s41596-020-0322-8)
* [Rodionova, O. Ye, and A. L. Pomerantsev. "Chemometric tools for food fraud detection: The role of target class in non-targeted analysis." Food chemistry 317 (2020): 126448.](https://www.sciencedirect.com/science/article/pii/S0308814620303101?casa_token=leLkME6puuUAAAAA:zVpftqGoeRPOrPQe3kC8lXb0SVD92sOJKvSD9hdcSyICKTACm77-GvTtrLmq4PMxBR_pF2oVxIw)
* [Pomerantsev, Alexey L., and Oxana Ye Rodionova. "Multiclass partial least squares discriminant analysis: Taking the right way—A critical tutorial." Journal of Chemometrics 32.8 (2018): e3030.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3030)
* [Oliveri, Paolo. "Class-modelling in food analytical chemistry: development, sampling, optimisation and validation issues–a tutorial." Analytica chimica acta 982 (2017): 9-19.](https://www.sciencedirect.com/science/article/pii/S0003267017306050?casa_token=oVsuIrrNPVwAAAAA:MEUtEmvrdm9s2He633MDjYEL9HhZ8rDewN3zCPWfVW8XkIfaS388sp1xkONVsWN6RcBR0EGdi6Y)
* [Rodionova, Oxana Ye, Anna V. Titova, and Alexey L. Pomerantsev. "Discriminant analysis is an inappropriate method of authentication." TrAC Trends in Analytical Chemistry 78 (2016): 17-22.)](https://www.sciencedirect.com/science/article/pii/S0165993615302193?casa_token=LhD_JTNn8PwAAAAA:9AsgZk7HsxoB8BDI88jZtNb8rbe48CiGT_lqtl8_RkF4EYABv3oltVi4N5YXe-CqRzvz3J_14bc)
* [Marini, Federico. "Classification methods in chemometrics." Current Analytical Chemistry 6.1 (2010): 72-79.](https://www.researchgate.net/profile/Federico-Marini-2/publication/232696878_Classification_methods_in_chemometrics/links/53fda10c0cf2364ccc08e208/Classification-methods-in-chemometrics.pdf)
* [Forina, M., et al. "Class-modeling techniques, classic and new, for old and new problems." Chemometrics and Intelligent Laboratory Systems 93.2 (2008): 132-148.](https://www.sciencedirect.com/science/article/pii/S0169743908000920?casa_token=teZELafmfmMAAAAA:evqDipEdosbDp2d6dmSXl4_eRbafJtY-KkzQgpjZhVc-VooGnXRvqAla91RBmJBriFhM7d5j7BQ)

## Topological Methods [Moderate amount of data available]
> "Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data. Though supervised variants exist, the typical manifold learning problem is unsupervised: it learns the high-dimensional structure of the data from the data itself, without the use of predetermined classifications." - sklearn [documentation](https://scikit-learn.org/stable/modules/manifold.html)
* [Kernel PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html?highlight=kernel%20pca#sklearn.decomposition.KernelPCA)
* [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap)
* [Locally Linear Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding)
* [Kohonen Self-Organizing Maps (SOM)](https://pypi.org/project/sklearn-som/) <!-- https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/-->
* [UMAP](https://umap-learn.readthedocs.io/en/latest/)

> These approaches may be considered intermediate between conventional methods and modern AI/ML algorithms.  These are generally non-linear dimensionality reduction methods that try to preserve properties, like the topology, of the original data; once projected into a lower dimensional space (embedding), statistical models can be constructed, for example, drawing an ellipse around the points belonging to a known class. Conventional chemometric authentication methods operate in a similar fashion but with a simpler dimensionality reduction step. Although [many methods](https://scikit-learn.org/stable/modules/outlier_detection.html) can be used to detect anomalies in this embedding (score space), we favor the elliptical envelope here for its simplicity and statistical interpretability. This approach is semi-supervised, since only members of one known class are purposefully trained on (at a time).
* [EllipticManifold](manifold/elliptic.py) - a combined manifold learning/dimensionality reduction step followed by the determination of a elliptical boundary.

<!-- scikit-tda; also PINN or PIML basically uses physics to constrain AI/ML to a manifold that respects physical constraints -->

## General Machine Learning [Large amount of data available]
> These routines offer the most flexible approaches and include alternative boundary construction methods besides ellipses.
* Outlier detection with [PyOD](https://pyod.readthedocs.io/en/latest/) - This encompasses many different approaches including isolation forests and autoencoders.
* Semi-supervised [Positive and Unlabeled (PU) learning](https://pulearn.github.io/pulearn/)

<!--
## Ensemble Models
> In machine learning, an ensemble model usually refers to a combination of (usually weaker) models that perform the same prediction task. Here, we use the term to refer to the combination of models that perform different tasks, i.e., each model is trained to predict the inlier/outlier status of a point with respect to one class.  We may combine such models so that a final prediction for an observation may be that it belongs to one class, many classes, or no known classes.  Efficiency, specificity and other metrics can then be computed from this.
-->

## Explanations and Interpretations
> While examination of loadings, for example, is one way to understand commonly employed chemometric tools, more complex models require more complex tools to inspect these "black boxes".
* [SHAP](https://shap.readthedocs.io/en/latest/) - "(SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions."

> "Interpretable AI" refers to models which are inherently a "glassbox" and their inner-workings are transparent.  This is [not the same as "explained black boxes" (XAI)](https://projecteuclid.org/journals/statistics-surveys/volume-16/issue-none/Interpretable-machine-learning-Fundamental-principles-and-10-grand-challenges/10.1214/21-SS133.full) which are inscrutable, by definition, but methods like SHAP can be used to help the user develop a sense of (dis)trust about the model and potentially debug it.  Explainable boosting machines (EBM) are (at the time of writing) a discrimination method, but can be helpful to compare and contrast with explained black boxes or authentication models. EBMs are slow to train so they are best for small-medium data applications, which many chemometric applications fall under.
* An [EBM](https://interpret.ml/docs/ebm.html) from [interpretML](https://interpret.ml) is a "tree-based, cyclic gradient boosting Generalized Additive Model with automatic interaction detection. EBMs are often as accurate as state-of-the-art blackbox models while remaining completely interpretable."
* [pyGAM](https://pygam.readthedocs.io/en/latest/index.html) does not follow scikit-learn's API but are very useful glassbox models to consider.

## Diagnostics
* [Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve) - these can be used to tell if you model will benefit from more data, or if you need a better model.

# Installation

~~~ bash
$ git clone https://github.com/mahynski/chemometrics.git
$ cd chemometrics
$ pip install -r requirements.txt
# If you wish to contribute to the repo you will need to install pre-commit as well
$ pip install pre-commit # (OPTIONAL)
~~~

Simply add this directory to your PYTHONPATH, or locally in each instance (i.e., sys.path.append()) and import the model as usual.

~~~ bash
$ echo 'export PYTHONPATH=$PYTHONPATH:/path/to/module/' >> ~/.bashrc
$ source ~/.bashrc
~~~

~~~ python
import chemometrics
~~~

You can run unittests to make sure your installation is working correctly.

~~~ bash
$ python -m unittest discover tests/
~~~

# Usage
Refer to `examples/` for example usage and more explicit details; you can [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahynski/chemometrics/blob/master/) to explore.

## Example Pseudocode
~~~ python
>>> from chemometrics.classifier.plsda import PLSDA
>>> X_train, X_test, y_train, y_test = load_data(...)
>>> sp = PLSDA(n_components=30, style='soft')
>>> _ = sp.fit(X_train.values, y_train.values)
>>> pred = sp.predict(X_train.values)
>>> df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = sp.figures_of_merit(pred, y_train.values)
~~~

## Deploying on Google Colab
You can use this repo in the cloud by using [Google Colab](https://colab.research.google.com).
Follow the instructions to set up an account if you do not already have one.

![](examples/colab_example.gif)

Below is the code that accompanies the gif above.

~~~python
# 1. Upload your data as a .csv file (enter this code and click "Choose Files")

from google.colab import files
uploaded = files.upload() # Currently there are some issues with this on Firefox

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
~~~

~~~python
# 2. Read your csv data into a Pandas DataFrame
import pandas as pd
df = pd.read_csv(list(uploaded.keys())[0])
~~~

~~~python
# Clone PyChemAuth repo
!git clone https://github.com/mahynski/PyChemAuth.git
!cd PyChemAuth; pip3 install -r requirements.txt
~~~~

~~~python
import PyChemAuth

# Perform analysis ...
~~~

# Other Tools
Other tools used in this repository include:
* [scikit-learn](https://scikit-learn.org/stable/)
* [Pandas](https://pandas.pydata.org/)
* [Bokeh](https://docs.bokeh.org/en/latest/)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Jupyter Notebooks](https://jupyter.org/)

# Citation

If you use UMAP refer to the authors' [github repo](https://github.com/lmcinnes/umap) for information about citation.  At the very least, you should cite the manuscript associated with the software itself:

``` bibtex
@article{mcinnes2018umap-software,
  title={UMAP: Uniform Manifold Approximation and Projection},
  author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
  journal={The Journal of Open Source Software},
  volume={3},
  number={29},
  pages={861},
  year={2018}
}
```

If you use [PyOD](https://pyod.readthedocs.io/en/latest/) be sure to cite:

``` bibtex
@article{zhao2019pyod,
  author  = {Zhao, Yue and Nasrullah, Zain and Li, Zheng},
  title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
  journal = {Journal of Machine Learning Research},
  year    = {2019},
  volume  = {20},
  number  = {96},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v20/19-011.html}
}
```

Refer to several citations for [SHAP](https://github.com/slundberg/shap) on the authors' website, but at a minimum be sure to cite:

``` bibtex
@incollection{NIPS2017_7062,
  title = {A Unified Approach to Interpreting Model Predictions},
  author = {Lundberg, Scott M and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems 30},
  editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
  pages = {4765--4774},
  year = {2017},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}
}
```

The [imbalanced-learn](https://imbalanced-learn.org/stable/index.html) package should be cited as:

``` bibtex
@article{JMLR:v18:16-365,
  author  = {Guillaume  Lema{{\^i}}tre and Fernando Nogueira and Christos K. Aridas},
  title   = {Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  volume  = {18},
  number  = {17},
  pages   = {1-5},
  url     = {http://jmlr.org/papers/v18/16-365.html}
}
```

 Refer to the [PU Learn](https://github.com/pulearn/pulearn) website for citation and credit attribution for positive and unlabeled learning.

 Refer to the [sklearn-som](https://sklearn-som.readthedocs.io/en/latest/) website for citation and credit attribution for Kohonen Self-Organizing Maps.
