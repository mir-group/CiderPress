# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CiderPress'
copyright = '2024, Kyle Bystrom'
author = 'Kyle Bystrom'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'breathe',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'

breathe_projects = {"CiderPress": "xml"}
breathe_default_project = "CiderPress"
breathe_default_members = ("members", "undoc-members")
breathe_domain_by_extension = {
    "h" : "c",
}

napoleon_google_docstring = True
# napoleon_numpy_docstring = True
napoleon_use_param = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "logos/cider_logo_and_name.png"
html_theme_options = {'logo_only': True}

