.. PyChemAuth documentation master file, created by
   sphinx-quickstart on Tue Aug 22 21:08:52 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
Python-based Chemometric Authentication
=======================================
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/
.. image:: https://github.com/mahynski/pychemauth/actions/workflows/python-app.yml/badge.svg?branch=main
   :target: https://github.com/mahynski/pychemauth/actions
.. image:: https://zenodo.org/badge/331207062.svg
   :target: https://zenodo.org/badge/latestdoi/331207062

This is a toolkit to perform chemometric authentication.  These methods are designed to follow `scikit-learn's estimator API <https://scikit-learn.org/stable/developers/develop.html>`_ so that they can be deployed in pipelines used with GridSearchCV, etc. and are compatible with workflows involving other modern machine learning tools.  Authentication is typically a `one-class classification (OCC) <https://en.wikipedia.org/wiki/One-class_classification>`_, or class modeling, approach designed to detect anomalies. This contrasts with multi-class classification (discriminative) models which involve supervised learning of multiple classes to distinguish between them; the primary weakness of this is that such a model typically cannot predict if a new sample belongs to **none** of the classes trained on.

.. image:: ../pychemauth.png
   
Within the context of anomaly detection, `scikit-learn <https://scikit-learn.org/stable/modules/outlier_detection.html>`_ differentiates between outlier detection and novelty detection.  In outlier detection, the training data is considered polluted and certain samples need to be detected and removed, whereas novelty detection methods assume the training data is "clean" and anomalies need to be detected during the testing phase of new samples only.  Both are important in the context of authentication models; this is a nice resource for a summary of `anomaly detection resources <https://github.com/yzhao062/anomaly-detection-resources>`_.

Out-of-distribution (OOD) detection is a more general term which encompasses these and other tasks, such as open-set recognition.  A taxonomy describing how these tasks are interrelated can be found `here <https://arxiv.org/abs/2110.11334>`_ and further reading `here <https://arxiv.org/abs/2110.14051>`_.


License Information
###################
* See LICENSE for more information.
* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by `NIST <https://www.nist.gov/>`_.

Core Capabilities
#################

Exploratory Data Analysis
*************************
You should always perform `exploratory data analysis <https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm>`_ to understand your data.  For example, understanding missing values, NaN, inf and basic `descriptive statistics <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html>`_.  The included `eda` module contains some additional tools for this.

Preprocessors
*************
`scikit-learn <https://scikit-learn.org>`_ provides a number of other simple `preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html>`_ steps, including data standardization and imputation approaches.  Here, these are extended to include:

Imputing Missing Data
=====================
* `Expectation Maximization with Iterative PCA (missing X values) <https://www.sciencedirect.com/science/article/pii/S0169743901001319?casa_token=PJMbl_1gHmoAAAAA:0Q4M969UyZ-MYQY44S0dFMtH77aX-AOxcCRSFBaDHuvsd2UnulLO3cUxh5GlHXnyJBzSp3oneO00>`_
* `Expectation Maximization with Iterative PLS (missing X values) <https://www.sciencedirect.com/science/article/pii/S0169743901001319?casa_token=PJMbl_1gHmoAAAAA:0Q4M969UyZ-MYQY44S0dFMtH77aX-AOxcCRSFBaDHuvsd2UnulLO3cUxh5GlHXnyJBzSp3oneO0>`_
* Limit of Detection (randomly below LOD)

Scaling
=======
* Corrected Scaling (akin to scikit-learn's `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_ but uses `unbiased/corrected standard deviation <https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation>`_ instead)
* Pareto Scaling (divides by square root of standard deviation)
* Robust Scaling (divides by IQR instead of standard deviation)

Filtering
=========
* `Savitzky-Golay <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_
* Standard and Robust Normal Variates, `SNV, RNV <https://www.sciencedirect.com/topics/mathematics/standard-normal-variate>`_
* Multiplicative Scatter Correction, `MSC <https://guifh.github.io/RNIR/MSC.html>`_

Generating Synthetic Data
=========================
* `Resampling <https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample>`_ can be used to balance classes during training, or to supplement measurements that are very hard to make.  New, synthetic data can also be generated by various means; `imblearn` pipelines are designed to work with various up/down sampling routines and can be used as drop-in replacements for standard scikit-learn pipelines.
* `Imbalanced Learning <https://imbalanced-learn.org/stable/index.html>`_ - SMOTE, ADASYN, etc.

Feature Selection
=================
`Feature extraction <https://scikit-learn.org/stable/modules/feature_extraction.html>`_, such as PCA, involves manipulating inputs to produce new "dimensions" or composite features, such as the first principal component. `Feature selection <https://scikit-learn.org/stable/modules/feature_selection.html>`_ simply involves selecting a subset of known features (such as columns) to use.  scikit-learn has many `built-in examples <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection>`_ that you can use.  Additional tools such as `BorutaSHAP <https://github.com/Ekeany/Boruta-Shap>`_ and some based on the `Jensen-Shannon Divergence <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_ are also implemented here.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
