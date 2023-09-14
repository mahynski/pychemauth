Quick Start Guide
=================

Installation
############

The easiest way to install pychemauth is with `pip <https://pypi.org/project/pip/>`_:

.. code-block:: python
   :linenos:
   
   pip install git+https://github.com/mahynski/pychemauth@main

You can also download the `repository <https://github.com/mahynski/pychemauth>`_ and install it from there. Check for the most recent version, or whichever is desired, and replace "X.X.X" below.

.. code-block:: bash
   :linenos:

   git clone https://github.com/mahynski/pychemauth.git --branch vX.X.X --depth 1
   cd pychemauth
   pip install .
   pytest # Optional, but recommended to run unittests

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
Start by creating a new notebook directly from inside your Google Drive account and proceed as follows.

.. image:: _static/colab_example.gif

Below is the code that accompanies the gif above.

How can you upload your data so you can access it from your Drive? There are 2 options.  The first involves
uploading the file(s) directly to this particular runtime instance.  If you are concerned about controlling
where this data goes, consider the second option of mounting your Drive from the notebook.  In that case you
can upload your data to your Drive and simply access it from there.

.. code-block:: python
   :linenos:

   # 1. Upload your data as a .csv file (enter this code and click "Choose Files")
   from google.colab import files
   my_file = files.upload() # Currently there are some issues with this on Firefox

   for fn in my_file.keys():
   print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(my_file[fn]))
   )

   # Read your csv data into a Pandas DataFrame
   import pandas as pd
   df = pd.read_csv(list(uploaded.keys())[0])


.. code-block:: python
   :linenos:

   # 2. Put the file in your Google Drive and access it from there
   from google.colab import drive
   drive.mount('/content/drive')

   # Your Drive is mounted here
   %ls drive/MyDrive/

   # Read your csv data into a Pandas DataFrame
   import pandas as pd
   df = pd.read_csv("/drive/MyDrive/my_file.csv")


You can then install pychemauth and begin your analysis.

.. code-block:: python
   :linenos:

   # 3. Install PyChemAuth
   !pip install git+https://github.com/mahynski/pychemauth@main
   import pychemauth

   # Perform analysis ...
