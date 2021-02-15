# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from operator import attrgetter
import inspect

sys.path.insert(0, os.path.abspath('../../tsanalysis'))

def linkcode_resolve(domain, info):
    """Resolve source code linkage.
    """

    if domain not in ('py', 'pyx'):
        return
    if not info.get('module') or not info.get('fullname'):
        return

    class_name = info['fullname'].split('.')[0]
    module = __import__(info['module'], fromlist=[class_name])
    obj = attrgetter(info['fullname'])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
            lineno = inspect.getsourcelines(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        print("Error in linkcode_resolve with domain: %s and info: %s", domain, info)
        resolved_link = 'https://github.com/ml-ngnm/time-series-analysis/'
    else:
        index_start = fn.index('tsanalysis')
        resolved_link = 'https://github.com/ml-ngnm/time-series-analysis/tree/develop/' + \
            fn[index_start:].replace('\\', '/') + '#L' + str(lineno)
    return resolved_link


# -- Project information -----------------------------------------------------

project = 'tsanalysis'
copyright = '2021, ml-ngnm'
author = 'Etienne de Montalivet'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     'sphinx.ext.napoleon',
#     'sphinx.ext.autodoc',
#     'numpydoc',
#     'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
#     'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
#     #'sphinx_autodoc_typehints',  # Automatically document param types (less noise in class signature)
#     #'sphinx.ext.linkcode',
#     'sphinx_rtd_theme',
#     'sphinx_gallery.gen_gallery',
# ]
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.linkcode',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgconverter',
    'sphinx_gallery.gen_gallery',
    # 'sphinx_issues',
#    'add_toctree_functions',
    'sphinx-prompt',
]

# numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

autodoc_default_options = {
    'members': True,
    'inherited-members': True
}

templates_path = ['templates']

# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True

add_function_parentheses = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'pydata_sphinx_theme' # sphinx_rtd_theme'
html_theme_path = ['themes']
html_theme = 'scikit-learn-modern'
# html4_writer=True
html_short_title = 'tsanalysis'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}