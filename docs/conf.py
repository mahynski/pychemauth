# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../pychemauth/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyChemAuth'
copyright = '2023, Nathan A. Mahynski'
author = 'Nathan A. Mahynski'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
        'sphinx.ext.napoleon',
        'sphinx.ext.mathjax',
        'sphinx_search.extension',
        'nbsphinx',
        'sphinx_gallery.load_style',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'jupyter']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
pygments_style = 'sphinx'
nbsphinx_execute = 'never' # Always run notebooks and save their output - just display
# nbsphinx_kernel_name = 'pychemauth-kernel'

nbsphinx_thumbnails = {
        'jupyter/gallery/simca_example': 'gallery/thumbs/default.png',
        'jupyter/gallery/imputing_examples': 'gallery/thumbs/default.png',
        'jupyter/gallery/pca_example': 'gallery/thumbs/default.png',
        'jupyter/gallery/pls_example': 'gallery/thumbs/default.png',
        'jupyter/gallery/plsda_example': 'gallery/thumbs/default.png',
        'jupyter/gallery/simca_example': 'gallery/thumbs/default.png',
        'jupyter/gallery/elliptic_manifold_example': 'gallery/thumbs/default.png',
        'jupyter/gallery/shap_example': 'gallery/thumbs/default.png',
}
