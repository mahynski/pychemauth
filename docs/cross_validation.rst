Cross Validation
================

`Cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ can be used to optimize optimize models and pipelines by providing a reasonable estimate of how they will perform on unseen data.  There are several versions of cross validation, some of which many be more optimal in certain situations than others.  In addition, the results from CV can be used to perform hypothesis testing to compare models or pipelines against each other to see if there is, in fact, any statistical significance to differences in performance.  This can be important since it can often be advantageous to choose the simplest, explainable model over the most complex, possible opaque one.  Understanding the tradeoff, if any, in performance is important to making this decision.

Here are some links to other resources:

* `"Model selection and overfitting," Lever, J., Krzywinski, M., Altman, N., Nature Methods 13, 703-704 (2016). <https://www.nature.com/articles/nmeth.3968.pdf>`_

* `"Cross-Validation," The Pennsylvania State University, STAT 555 (2018). <https://online.stat.psu.edu/stat555/node/118/>`_

In the examples provided here, we will assume all data is `independent and identically distributed <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`_ (IID).  This means there is no internal structure to the data so we can use basic k-fold or other simplistic data splits for cross-validation.  However, if you have multiple measurements that are part of a group you should probably use a "group" variant (e.g., GroupKFold vs. KFold) instead.  For example, if you have a dataset where multiple measurements come from the same person, place, or year in a dataset that spans multiple people, places, or times.  These "group" variants ensure that when one group is placed in the training set it does not appear in the test set; thus, data does not "leak" between these folds, which would otherwise lead to an overly optimistic model. Refer to the documentation in the `compare <https://pychemauth.readthedocs.io/en/latest/pychemauth.analysis.html#pychemauth.analysis.compare.Compare>`_ submodule for more information.

For regression problems, the `Kennard-Stone <https://pypi.org/project/kennard-stone/>`_ algorithm might also be useful for creating balanced splits.  This and other "rational" data splits have been implemented in a scikit-learn-compatible python library called `"astartes" <https://github.com/JacksonBurns/astartes>`_ which you can also refer to.  PyChemAuth currently makes use of various data-splitters available in scikit-learn (summarized `here <https://scikit-learn.org/dev/api/sklearn.model_selection.html>`_) and provides KFold Kennard-Stone via `this <https://pypi.org/project/kennard-stone/>`_ package.  Additional compatibility may be provided in the future.

.. nbgallery::
   :hidden:

   jupyter/learn/cv_optimization
   jupyter/learn/cv_comparison
