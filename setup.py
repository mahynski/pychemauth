"""
Install pychemauth.

author: nam
"""
import setuptools

# Get __version__ from __init__ file
exec(open("pychemauth/__init__.py").read())

setuptools.setup(
    name="pychemauth",
    description="Python-based Chemometric Authentication",
    author="Nathan A. Mahynski",
    homepage="https://github.com/mahynski/pychemauth",
    python_requires=">=3.10.0",
    version=__version__,
    packages=setuptools.find_packages(),
    license_files=("LICENSE",),
    test_suite="tests",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "kennard-stone==2.2.1",
        "mypy==1.11.2",  # For type hints and checking
        "types-requests==2.32.0.20240914",  # For type checking
        "torch==2.4.1+cpu",  # For DIME - just need torch tensor libraries so no need for GPU support which can also cause conflicts
        "dime-pytorch==1.0.1",
        "baycomp==1.0.3",
        "bokeh>=3.4.3",  # Default in Colab
        "bokeh_sampledata==2024.2",
        "BorutaShap @ git+https://github.com/Ekeany/Boruta-Shap.git@38af879",
        "imbalanced-learn==0.11.0",
        "IPython",
        "ipywidgets",
        "matplotlib==3.7.2",
        "nodejs==0.1.1",
        "numpy==1.26.4",
        "pandas==2.1.4",  # For Colab
        "pre-commit==3.3.3",
        "scikit-learn==1.3.0",
        "scipy==1.11.1",
        "seaborn==0.12.2",
        "shap==0.45.1",
        "tqdm==4.66.1",
        "umap-learn==0.5.3",
        "watermark==2.4.3",
        "pytest==7.4.0",
        "xgboost==2.0.0",
        "missingno==0.5.2",
        "wandb>=0.17.5",
        "pyts==0.13.0",
        "pillow>=10.0.0",
        "visualkeras>=0.1.3",
        "opencv-python==4.10.0.82",  # For SHAP explanations
        "huggingface_hub==0.23.4",
        "tensorflow-cpu==2.14.0",  # This command should install keras==2.14.0 as well - based on Keras recommendation (https://keras.io/getting_started/#installing-keras-3) for creating a "universal GPU environment" based on Colab recommendations: https://colab.research.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu - CPU only default is more portable, we can overwrite on Colab
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cpu",  # For torch
    ],
    extras_require={
        "gpu": ["tensorflow[and-cuda]==2.14.0"]
    }
)
