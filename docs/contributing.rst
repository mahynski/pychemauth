Contributing
============

If you find a bug, please `submit a bug report <https://github.com/mahynski/pychemauth/issues/new/choose>`_ using the pre-configured template on the repository.  In your report, please include:

1. A minimum amount of code to reproduce the error from scratch (including raw data), and
2. The anticipated output which differs from what you were expecting.

You put also make `feature requests <https://github.com/mahynski/pychemauth/issues/new/choose>`_ using the pre-configured template on the repository.

Community contributions to pychemauth are also welcome.  If you would like to contribute, follow the recommended `GitHub workflow <https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>`_:

1. Fork the project.
2. Create a topic branch from master.
3. Make some commits to improve the project.
4. Push this branch to **your** GitHub project.
5. Open a Pull Request on the pychemauth GitHub repo.
6. Discuss, and optionally continue committing.
7. The pychemauth maintainers will merges or close the Pull Request.
8. Sync the updated pychemauth master back to your fork.

Code Standards
##############

Linting
*******

Please note, style and code linting is enforce with `pre-commit <https://pre-commit.com/>`_.  Refer to the .pre-commit-config.yaml file in the `GitHub repo <https://github.com/mahynski/pychemauth>`_ for specific details.  
Here is some pseudocode illustrating how to use this after you (USERNAME) have created your own fork of pychemauth.

.. code-block:: bash

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


