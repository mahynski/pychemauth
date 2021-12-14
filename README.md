chemometrics
============
Common chemometric analysis methods implemented in python.  These methods are designed to be compatible with [scikit-learn's estimator API](https://scikit-learn.org/stable/developers/develop.html) so they can be deployed in pipelines used with GridSearchCV, etc.

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/mahynski/chemometrics/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/chemometrics)
[![Build Status](https://travis-ci.com/mahynski/chemometrics.svg?branch=main)](https://travis-ci.com/mahynski/chemometrics)

## Installation

~~~ bash
$ git clone https://github.com/mahynski/chemometrics.git
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

## Unittests
~~~ bash
$ python -m unittest discover tests/
~~~

## Available Classifiers
* PLS-DA (soft and hard variants)
* SIMCA
* DD-SIMCA

## Available Regressors (To Do)
* PCA
* PCR
* PLS(2) Regression

## Example Usage
~~~ python
>>> from chemometrics.classifier.plsda import PLSDA
>>> X_train, X_test, y_train, y_test = load_data(...)
>>> sp = PLSDA(n_components=30, style='soft')
>>> _ = sp.fit(X_train.values, y_train.values)
>>> pred = sp.predict(X_train.values)
>>> df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = sp.figures_of_merit(pred, y_train.values)
~~~

## Deploying on Google Colab
