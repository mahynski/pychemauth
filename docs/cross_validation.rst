Cross Validation
================

[Cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) can be used to optimize optimize models and pipelines by providing a reasonable estimate of how they will perform on unseen data.  There are several versions of cross validation, some of which many be more optimal in certain situations than others.  In addition, the results from CV can be used to perform hypothesis testing to compare models or pipelines against each other to see if there is, in fact, any statistical significance to differences in performance.  This can be important since it can often be advantageous to choose the simplest, explainable model over the most complex, possible opaque one.  Understanding the tradeoff, if any, in performance is important to making this decision.

.. nbgallery::
   :hidden:

   jupyter/gallery/cv_optimization
   jupyter/gallery/cv_comparison
