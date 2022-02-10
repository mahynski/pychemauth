Chemometric Authentication
==========================
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![Workflow](https://github.com/mahynski/chemometrics/actions/workflows/python-app.yml/badge.svg?branch=main)
<!--[![codecov](https://codecov.io/gh/mahynski/chemometrics/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/chemometrics)-->

This is a centralized repository of common (and emerging) tools implemented in python to perform chemometric authentication.  These methods are designed to be compatible with [scikit-learn's estimator API](https://scikit-learn.org/stable/developers/develop.html) so they can be deployed in pipelines used with GridSearchCV, etc.  Authentication is typically a [one-class classification (OCC)](https://en.wikipedia.org/wiki/One-class_classification), or class modeling, problem designed to detect anomalies.

> "Outlier detection and novelty detection are both used for anomaly detection, where one is interested in detecting abnormal or unusual observations. Outlier detection is then also known as unsupervised anomaly detection and novelty detection as semi-supervised anomaly detection. In the context of outlier detection, the outliers/anomalies cannot form a dense cluster as available estimators assume that the outliers/anomalies are located in low density regions. On the contrary, in the context of novelty detection, novelties/anomalies can form a dense cluster as long as they are in a low density region of the training data, considered as normal in this context." - [scikit-learn's documentation](https://scikit-learn.org/stable/modules/outlier_detection.html)

Essentially, outlier detection methods characterize inliers as those points in high density regions, whereas novelty detection routines try to characterize a boundary around the region where a known class is found (even if it disperse). Both can be useful when attempting to detect chemometric anomalies. 

## License Information
* See LICENSE for more information.
* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by [NIST](https://www.nist.gov/).

# Installation

~~~ bash
$ git clone https://github.com/mahynski/chemometrics.git
# cd chemometrics
$ pip install -r requirements.txt
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

# Capabilities

## Preprocessors
[scikit-learn](https://scikit-learn.org) provides a number of other simple [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) steps, including data standardization and imputation approaches.  Here, these are extended to include:

### Imputing Missing Data
* Expectation Maximization with Iterative PCA (missing X values)
* Expectation Maximization with Iterative PLS (missing X values)
* Limit of Detection (randomly below LOD)

### Scaling
* Corrected Scaling (akin to scikit-learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) but uses [unbiased/corrected standard deviation](https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation) instead)
* Pareto Scaling (scales by square root of standard deviation)
* Robust Scaling (scales by IQR instead of standard deviation)

<!--
### Generating Synthetic Data
Can be used to balance classes during training, or to supplement measurements that are very hard to make.
* [Imbalanced Learning](https://imbalanced-learn.org/stable/index.html) - SMOTE, ADASYN, etc.
* Generative Networks (VAE, GAN)
-->

## Conventional Chemometrics [Least amount of data available]

> Conventional chemometric authentication methods generally fall under the umbrella of multivariate regression or classification tasks.  For example, the model proposed when performing multilinear regression is  `y = MX + b`, where the matrix `M` must be solved for.  (Un)supervised classification is commonly performed via projection methods, which create a model of the data as: `X = TP^T + E`, where the scores matrix, `T`, represents the projection of the `X` matrix into a (usually lower dimensional) score space. The `P` matrix, called the [loadings matrix](http://www.statistics4u.com/fundstat_eng/cc_pca_loadscore.html), is computed in different ways.  For example, PCA uses the leading eigenvectors of the covariance matrix of `X`, where as PLS uses a different (supervised) decomposition which is a function of both `X` and `y`. `E` is the error resulting from this model.

> These often require careful preparation of the training set to remove extremes and outliers so that "masking" effects do not affect your final model. Manual data inspection is typically (but not always) required, whereas many machine learning-based outlier detection methods are robust against outliers natively. Thus, conventional authentication methods can be considered [novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html) methods (no outliers in training), but many have built in capabilities to interatively "clean" the training set if outliers are assumed to be present initially. See ["Detection of Outliers in Projection-Based Modeling" by Rodionova and Pomerantsev](https://pubs.acs.org/doi/abs/10.1021/acs.analchem.9b04611) for an example of outlier detection and removal in projection-based modeling.

### Classifiers
* PCA (for data inspection)
* PLS-DA (soft and hard variants) - [discriminant analysis is not the same as OCC for authentication](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/cem.3030).
* SIMCA
* DD-SIMCA

### Regressors
* PCR
* PLS

### Non-parametric Methods
* SRD (Sum of Ranking Differences)

## Manifold Learning [Moderate amount of data available]
> "Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data. Though supervised variants exist, the typical manifold learning problem is unsupervised: it learns the high-dimensional structure of the data from the data itself, without the use of predetermined classifications." - sklearn [documentation](https://scikit-learn.org/stable/modules/manifold.html)
* [Kernel PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html?highlight=kernel%20pca#sklearn.decomposition.KernelPCA)
* [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap)
* [Locally Linear Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding)
* [UMAP](https://umap-learn.readthedocs.io/en/latest/)

> These approaches may be considered intermediate in complexity between conventional, projection-based methods and modern AI/ML algorithms.  These are generally non-linear dimensionality reduction methods that try to preserve properties, like the topology, of the original data; once projected into a lower dimensionsal space, statistical models can be constructed, for example, drawing an ellipse around the points belonging to a known class. Conventional methods operate in a similar fashion but with a simpler dimensionality reduction step. Although [many methods](https://scikit-learn.org/stable/modules/outlier_detection.html) can be used to detect anomalies in this score space, we favor the elliptical envelope here for its simplicity and statistical interpretability.
* EllipticalManifold - a combined manifold learning/dimensionality reduction step followed by the determination of a elliptical boundary.

## General Machine Learning [Large amount of data available]
> These routines offer the most flexible approaches and include alternative boundary construction methods besides ellipses.
* Outlier detection with [PyOD](https://pyod.readthedocs.io/en/latest/) - This encompasses many different approaches including isolation forests and autoencoders.
* Semi-supervised [Positive and Unlabeled (PU) learning](https://pulearn.github.io/pulearn/)

<!--
## Ensemble Models
> In machine learning, an ensemble model usually refers to a combination of (usually weaker) models that perform the same prediction task. Here, we use the term to refer to the combination of models that perform different tasks, i.e., each model is trained to predict the inlier/outlier status of a point with respect to one class.  We may combine such models so that a final prediction for an observation may be that it belongs to one class, many classes, or no known classes.  Efficiency, specificity and other metrics can then be computed from this.
-->

## Explanations
> While examination of loadings, for example, is one way to understand commonly employed chemometric tools, more complex models require more complex tools to inspect these "black boxes".
* [SHAP](https://shap.readthedocs.io/en/latest/)

## Diagnostics
* Learning curves - these can be used to tell if you model will benefit from more data, or if you need a better model.

# Usage 
Refer to `examples/` for example usage and more explicit details.

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
# Clone chemometrics repo
!git clone https://github.com/mahynski/chemometrics.git
!cd chemometrics; pip install -r requirements.txt
~~~~

~~~python
import chemometrics

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

 Refer the [PU Learn](https://github.com/pulearn/pulearn) website for citation and credit attribution.
