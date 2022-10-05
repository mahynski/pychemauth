"""
Install pychemauth.

@author: nam
"""
from setuptools import find_packages, setup

try:
    from version import __version__
except ModuleNotFoundError:
    exec(open("version.py").read())

setup(
    name="pychemauth",
    description="Python-based Chemometric Authentication",
    author="Nathan A. Mahynski",
    python_requires=">=3.7.0",
    version=__version__,
    packages=find_packages(),
    license_files=("LICENSE",),
    test_suite="tests",
    tests_require=["pytest"],
    install_requires=[
        "baycomp",
        "bokeh",
        "BorutaShap",
        "imbalanced-learn",
        "IPython",
        "ipywidgets",
        "matplotlib",
        "nodejs",
        "numpy",
        "pandas",
        "pre-commit",
        "scikit-learn",
        "scipy",
        "seaborn",
        "shap",
        "tqdm",
        "umap-learn",
        "watermark",
    ],
)
