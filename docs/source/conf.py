# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scmiracle'
copyright = '2025, labomics'
author = 'labomics'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys, os

docs_source_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(docs_source_dir, '../..'))
sys.path.insert(0, repo_root)

repository_url = "https://github.com/sc-miracle/miracle"

nb_execute_notebooks = "off"


extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.doctest',
   'sphinx.ext.intersphinx',
   'sphinx.ext.todo',
   'sphinx.ext.coverage',
   'sphinx.ext.mathjax',
   'sphinx.ext.napoleon',
   'sphinx.ext.ifconfig',
   'sphinx.ext.viewcode',
   'sphinx.ext.githubpages',
   'recommonmark',
   'sphinx_markdown_tables',
   'nbsphinx',
   'sphinx.ext.mathjax',
   ]
try:
   import sphinx_copybutton  # noqa: F401
except ModuleNotFoundError:
   pass
else:
   extensions.append('sphinx_copybutton')
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
from recommonmark.parser import CommonMarkParser

source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['api/modules.rst']

html_static_path = ['_static']
# html_logo = "_static/img/midas_small_color3.svg"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

try:
   import sphinx_book_theme  # noqa: F401
except ModuleNotFoundError:
   html_theme = 'alabaster'
else:
   html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
} if html_theme == 'sphinx_book_theme' else {}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
