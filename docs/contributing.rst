Contributing
============

Bugs
####

If you find a bug, please `submit a bug report <https://github.com/mahynski/pychemauth/issues/new/choose>`_ using the pre-configured template on the repository.  In your report, please include:

1. A minimum amount of code to reproduce the error from scratch (including raw data), and
2. The anticipated output which differs from what you were expecting.

New Features
############

You put also make `feature requests <https://github.com/mahynski/pychemauth/issues/new/choose>`_ using the pre-configured template on the repository.

Community Contributions
########################

Community contributions to PyChemAuth are also welcome.  If you would like to contribute, follow the recommended `GitHub workflow <https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>`_:

1. Fork the project.
2. Create a topic branch from master.
3. Make some commits to improve the project.
4. Push this branch to **your** GitHub project.
5. Open a Pull Request on the PyChemAuth GitHub repo.
6. Discuss, and optionally continue committing.
7. The PyChemAuth maintainers will merges or close the Pull Request.
8. Sync the updated PyChemAuth master back to your fork.

Code Standards
##############

Linting
*******

Please note, style and code linting is enforce with `pre-commit <https://pre-commit.com/>`_.  Refer to the .pre-commit-config.yaml file in the `GitHub repo <https://github.com/mahynski/pychemauth>`_ for specific details.  
Here is some pseudocode illustrating how to use this after you (USERNAME) have created your own fork of PyChemAuth.

.. code-block:: bash
   :linenos:

   git clone https://github.com/USERNAME/forked-pychemauth.git
   cd pychemauth
   pip install pre-commit
   pre-commit install
   # Make your changes
   ...
   pre-commit run --all-files
   git commit -m "I added some new features!" .

Documentation
*************

Code is documented using numpy docstrings.  This style is documented `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and an illustrative example is available `here <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.
Please document any changes and follow this convention when making contributions.

This uses reStructuredText for python:

* Documentation can be found `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain>`_.
* A nice `cheatsheet <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_.

Jupyter Notebooks
#################

Notebooks are a great way to illustrate the correct use of your code and facilitates its adoption.  We welcome contributed notebooks as well, subject to review. 

Use `pandoc flavored markdown <https://pandoc.org/MANUAL.html#pandocs-markdown>`_ comments in your notebook with a single level-one heading in the top cell to denote the notebook's name; e.g., <h1>DD-SIMCA Example</h1>.  A subsequent cell can be used to describe the notebook further.

.. code-block:: html
   :linenos:

   DD-SIMCA Example
   ===

.. code-block:: html
   :linenos:

   Author: Nathan A. Mahynski

   Date: 2023/08/23

   Description: ...

* Some good suggestions to improve readability are given `here <https://www.kaggle.com/code/alejopaullier/make-your-notebooks-look-better>`_.


