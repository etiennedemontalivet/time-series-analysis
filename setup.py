"""
Setup module for ai-analysis
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='AI analysis',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    version='0.1.0',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Machine Learning library for analysis',  # Optional

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/pypa/sampleproject',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='Etienne de Montalivet',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='etienne.demontalivet@gmail.com',  # Optional

    # Classifiers help users find your project by categorizing it.
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='ml datascience',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['tests', 'docs', 'notebooks', 'contribs', 'pip-wheel-metadata', 'venv']),  # Required

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. If you
    # do not support Python 2, you can simplify this to '>=3.5' or similar, see
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.6',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'lightgbm>=2.3.0',
        'numpy>=1.17.3',
        'pandas>=0.25.2',
        'pyentrp>=0.5.0',
        'PyWavelets>=1.0.6',
        'pyyaml>=5.1.2',
        'seaborn>=0.9.0',
        'matplotlib>=3.1.2',
        'scikit-optimize',
        'umap-learn==0.3.10',
        'shap==0.32.1',
        'h2o==3.26.0.9',
        'keras==2.2.4',
        'tensorflow==1.14.0',
        'plotly==4.3.0',
    	'imbalanced-learn==0.5.0',
        'imblearn==0.0',
        'autopep8==1.5',
        'optuna==1.1.0',
        'PyAstronomy==0.14.0',
        'jupytext==1.5.0'
        'jupyterlab'
    ],  # Optional
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install ai-analysis[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        'dev': [
            'flake8>=3.7',
            'black',
            'sphinx>=2.2',
            'invoke>=1.3',
            'pytest',
            'pytest-coverage'
        ],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={},
    data_files=[],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    entry_points={},

    # List additional URLs that are relevant to your project as a dict.
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={},
)
