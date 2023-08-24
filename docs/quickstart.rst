Quick Start Guide
=================

Installation
############

Vist the `github repo <https://github.com/mahynski/pychemauth>`_ to check for the most recent version and replace "X.X.X" below.

.. code-block:: bash
   :linenos:

   git clone https://github.com/mahynski/pychemauth.git --branch vX.X.X --depth 1
   cd pychemauth
   pip install .

You should run unittests to make sure your installation is working correctly.

.. code-block:: bash
   :linenos:

   pytest

Usage
#####

Simply import the package to begin using it.

.. code-block:: python
   :linenos:

   import pychemauth

Some example psuedocode might look like this:

.. code-block:: python
   :linenos:

   from pychemauth.classifier.plsda import PLSDA
   X_train, X_test, y_train, y_test = load_data(...)
   model = PLSDA(n_components=10, style='soft')
   model.fit(X_train.values, y_train.values)
   pred = model.predict(X_test.values)
   df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = model.figures_of_merit(pred, y_test.values)


Deploying on `Colab <https://colab.google/>`_
##############################################

You can use pychemauth in the cloud for free by using `Google Colab <https://colab.research.google.com>`_.
Click the link and follow the instructions to set up an account if you do not already have one.

.. image:: colab_example.gif

Below is the code that accompanies the gif above.

.. code-block:: python
   :linenos:

   # 1. Upload your data as a .csv file (enter this code and click "Choose Files")
   from google.colab import files
   uploaded = files.upload() # Currently there are some issues with this on Firefox

   for fn in uploaded.keys():
   print('User uploaded file "{name}" with length {length} bytes'.format(
    name=fn, length=len(uploaded[fn])))


.. code-block:: python
   :linenos:

   # 2. Read your csv data into a Pandas DataFrame
   import pandas as pd
   df = pd.read_csv(list(uploaded.keys())[0])


.. code-block:: python
   :linenos:

   # 3. Clone PyChemAuth repo
   !git clone https://github.com/mahynski/pychemauth.git --depth 1 --branch vX.Y.Z
   !cd pychemauth; pip install .; cd ..


.. code-block:: python
   :linenos:

   import pychemauth

   # Perform analysis ...
