Cross Validation
================

`Cross validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ can be used to optimize optimize models and pipelines by providing a reasonable estimate of how they will perform on unseen data.  There are several versions of cross validation, some of which many be more optimal in certain situations than others.  In addition, the results from CV can be used to perform hypothesis testing to compare models or pipelines against each other to see if there is, in fact, any statistical significance to differences in performance.  This can be important since it can often be advantageous to choose the simplest, explainable model over the most complex, possible opaque one.  Understanding the tradeoff, if any, in performance is important to making this decision.

Here are some links to other resources:

* `"Model selection and overfitting," Lever, J., Krzywinski, M., Altman, N., Nature Methods 13, 703-704 (2016). <https://www.nature.com/articles/nmeth.3968.pdf>`_

* `"Cross-Validation," The Pennsylvania State University, STAT 555 (2018). <https://online.stat.psu.edu/stat555/node/118/>`_

.. nbgallery::
   :hidden:

   jupyter/learn/cv_optimization
   jupyter/learn/cv_comparison
