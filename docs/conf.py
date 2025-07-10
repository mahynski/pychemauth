"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../pychemauth/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyChemAuth"
copyright = "2024, Nathan A. Mahynski"
author = "Nathan A. Mahynski"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_search.extension",
    "nbsphinx",
    "sphinx_gallery.load_style",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/logo_no.png"
html_theme = "sphinx_book_theme"  #'sphinx_rtd_theme'
html_static_path = ["_static"]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "mahynski", # Username
    "github_repo": "pychemauth", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}

pygments_style = "sphinx"
nbsphinx_execute = "never"  # Always pre-run notebooks and save their output - just display on readthedocs
# nbsphinx_kernel_name = 'pychemauth-kernel'
nbsphinx_thumbnails = {
    "jupyter/api/pipelines": "_static/default.png",
    "jupyter/api/sharing_models": "_static/default.png",
}
