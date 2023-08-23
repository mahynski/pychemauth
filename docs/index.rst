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

Within the context of anomaly detection, `scikit-learn <https://scikit-learn.org/stable/modules/outlier_detection.html>`_ differentiates between outlier detection and novelty detection.  In outlier detection, the training data is considered polluted and certain samples need to be detected and removed, whereas novelty detection methods assume the training data is "clean" and anomalies need to be detected during the testing phase of new samples only.  Both are important in the context of authentication models; this is a nice resource for a summary of `anomaly detection resources <https://github.com/yzhao062/anomaly-detection-resources>`_.

Out-of-distribution (OOD) detection is a more general term which encompasses these and other tasks, such as open-set recognition.  A taxonomy describing how these tasks are interrelated can be found `here] <https://arxiv.org/abs/2110.11334>`_ and further reading `here <https://arxiv.org/abs/2110.14051>`_.


License Information
###################
* See LICENSE for more information.
* Any mention of commercial products is for information only; it does not imply recommendation or endorsement by `NIST <https://www.nist.gov/>`_.

Core Capabilities
#################

.. image:: pychemauth.png

Exploratory Data Analysis
*************************


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
