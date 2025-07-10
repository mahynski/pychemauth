Citations
=========

If you use PyChemAuth in a publication, please cite the appropriate version (most recent is linked below).

.. image:: https://zenodo.org/badge/331207062.svg
   :target: https://zenodo.org/badge/latestdoi/331207062

PyChemAuth contains both original code and wrappers around other packages and thus relies on contributions from many other sources.
If you use these tools be sure to cite the original authors.

Code
####

If you use the Kennard-Stone features in PyChemAuth please cite the original authors:

.. code-block:: bibtex

   @misc{kennard-stone,
   title={kennard-stone},
   author={yu9824},
   year={2021},
   howpublished={\url{https://github.com/yu9824/kennard_stone}},
   }

If you use UMAP refer to the authors' `github repo <https://github.com/lmcinnes/umap>`_ for information about citation.
At the very least, you should cite the manuscript associated with the software itself:

.. code-block:: bibtex

   @article{mcinnes2018umap-software,
   title={UMAP: Uniform Manifold Approximation and Projection},
   author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
   journal={The Journal of Open Source Software},
   volume={3},
   number={29},
   pages={861},
   year={2018}
   }

If you use `PyOD <https://pyod.readthedocs.io/en/latest/>`_ be sure to cite:

.. code-block:: bibtex

   @article{zhao2019pyod,
   author  = {Zhao, Yue and Nasrullah, Zain and Li, Zheng},
   title   = {PyOD: A Python Toolbox for Scalable Outlier Detection},
   journal = {Journal of Machine Learning Research},
   year    = {2019},
   volume  = {20},
   number  = {96},
   pages   = {1-7},
   url     = {http://jmlr.org/papers/v20/19-011.html}
   }

Refer to several citations for `SHAP <https://github.com/slundberg/shap>`_ on the authors' website, but at a minimum be sure to cite:

.. code-block:: bibtex

   @incollection{NIPS2017_7062,
   title = {A Unified Approach to Interpreting Model Predictions},
   author = {Lundberg, Scott M and Lee, Su-In},
   booktitle = {Advances in Neural Information Processing Systems 30},
   editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
   pages = {4765--4774},
   year = {2017},
   publisher = {Curran Associates, Inc.},
   url = {http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf}
   }

The `imbalanced-learn <https://imbalanced-learn.org/stable/index.html>`_ package should be cited as:

.. code-block:: bibtex

   @article{JMLR:v18:16-365,
   author  = {Guillaume  Lema{{\^i}}tre and Fernando Nogueira and Christos K. Aridas},
   title   = {Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning},
   journal = {Journal of Machine Learning Research},
   year    = {2017},
   volume  = {18},
   number  = {17},
   pages   = {1-5},
   url     = {http://jmlr.org/papers/v18/16-365.html}
   }

If you use any `Keras <https://keras.io/>`_ models, be sure to cite:

.. code-block:: bibtex

   @misc{chollet2015keras,
   title={Keras},
   author={Chollet, Fran\c{c}ois and others},
   year={2015},
   howpublished={\url{https://keras.io}},
   }

PyChemAuth is configured to use the `tensorflow <>`_ backend of Keras, so if you use Keras please also cite:

.. code-block:: bibtex

   @misc{tensorflow2015-whitepaper,
   title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
   url={https://www.tensorflow.org/},
   note={Software available from tensorflow.org},
   author={
      Mart\'{i}n~Abadi and
      Ashish~Agarwal and
      Paul~Barham and
      Eugene~Brevdo and
      Zhifeng~Chen and
      Craig~Citro and
      Greg~S.~Corrado and
      Andy~Davis and
      Jeffrey~Dean and
      Matthieu~Devin and
      Sanjay~Ghemawat and
      Ian~Goodfellow and
      Andrew~Harp and
      Geoffrey~Irving and
      Michael~Isard and
      Yangqing Jia and
      Rafal~Jozefowicz and
      Lukasz~Kaiser and
      Manjunath~Kudlur and
      Josh~Levenberg and
      Dandelion~Man\'{e} and
      Rajat~Monga and
      Sherry~Moore and
      Derek~Murray and
      Chris~Olah and
      Mike~Schuster and
      Jonathon~Shlens and
      Benoit~Steiner and
      Ilya~Sutskever and
      Kunal~Talwar and
      Paul~Tucker and
      Vincent~Vanhoucke and
      Vijay~Vasudevan and
      Fernanda~Vi\'{e}gas and
      Oriol~Vinyals and
      Pete~Warden and
      Martin~Wattenberg and
      Martin~Wicke and
      Yuan~Yu and
      Xiaoqiang~Zheng},
   year={2015},
   }

If you use "DIME" to perform out-of-distribution detection on a neural network model, please cite:

.. code-block:: bibtex

   @misc{sjogren2021outofdistribution,
   title = {Out-of-Distribution Example Detection in Deep Neural Networks using Distance to Modelled Embedding},
   author = {Rickard Sjögren and Johan Trygg},
   year = {2021},
   eprint = {2108.10673},
   archivePrefix = {arXiv},
   primaryClass = {cs.LG}
   }

If you use `visualkeras <https://github.com/paulgavrikov/visualkeras>`_ to visualize any Keras models, please cite:

.. code-block:: bibtex

   @misc{Gavrikov2020VisualKeras,
   author = {Gavrikov, Paul},
   title = {visualkeras},
   year = {2020},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/paulgavrikov/visualkeras}},
   }

If you use `pyts <https://pyts.readthedocs.io/en/stable/index.html>`_ to "image" series, or in any other way, please cite:

.. code-block:: bibtex

   @article{JMLR:v21:19-763,
   author  = {Johann Faouzi and Hicham Janati},
   title   = {pyts: A Python Package for Time Series Classification},
   journal = {Journal of Machine Learning Research},
   year    = {2020},
   volume  = {21},
   number  = {46},
   pages   = {1-6},
   url     = {http://jmlr.org/papers/v21/19-763.html}
   }

Refer to the `PU Learn <https://github.com/pulearn/pulearn>`_ website for citation and credit attribution for positive and unlabeled learning.

Refer to the `sklearn-som <https://sklearn-som.readthedocs.io/en/latest/>`_ website for citation and credit attribution for Kohonen Self-Organizing Maps.

Data
####

Example data used in this repository comes from several sources; refer to the documentation for each data loader (e.g., :func:`load_pgaa`) for the appropriate citation(s).
