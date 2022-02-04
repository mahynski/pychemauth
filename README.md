chemometrics
============
Common chemometric analysis methods implemented in python.  These methods are designed to be compatible with [scikit-learn's estimator API](https://scikit-learn.org/stable/developers/develop.html) so they can be deployed in pipelines used with GridSearchCV, etc.

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/mahynski/chemometrics/branch/main/graph/badge.svg?token=YSLBQ33C7F)](https://codecov.io/gh/mahynski/chemometrics)
[![Workflow](https://github.com/mahynski/chemometrics/actions/workflows/python-app.yml/badge.svg?branch=main)]

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
* PCA (for data inspection)
* PLS-DA (soft and hard variants)
* SIMCA
* DD-SIMCA

## Available Regressors
* PCR
* PLS

## Example Usage (Pseudocode)
~~~ python
>>> from chemometrics.classifier.plsda import PLSDA
>>> X_train, X_test, y_train, y_test = load_data(...)
>>> sp = PLSDA(n_components=30, style='soft')
>>> _ = sp.fit(X_train.values, y_train.values)
>>> pred = sp.predict(X_train.values)
>>> df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = sp.figures_of_merit(pred, y_train.values)
~~~

Refer to `examples/` for example usage and more explicit details.

## Deploying on Google Colab

You can use this repo in the cloud by using [Google Colab](https://colab.research.google.com).
Follow the instructions to set up an account if you do not already have one.
Then, enter the following cells at the start of a new notebook:

~~~python
# 1. Upload your data as a .csv file

from google.colab import files
uploaded = files.upload() # Currently there are some issues with this on Firefox

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
~~~

~~~python
# 2. Read your csv data into a Pandas DataFrame - split into test/train later
import pandas as pd
df = pd.read_csv(io.StringIO(uploaded[''].decode('utf-8')), sep=';')
~~~

~~~python
# Clone chemometrics repo
!git clone https://github.com/mahynski/chemometrics.git
!pip install -r requirements.txt
~~~~

~~~python
import chemometrics

# Begin analysis ...
~~~
