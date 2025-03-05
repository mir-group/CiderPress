# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
from datetime import datetime
from ciderpress import __version__

# sys.path.insert(0, str(Path('..').resolve()))
sys.path.append("./tools/extensions")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CiderPress'
author = 'Kyle Bystrom'
year = datetime.now().year
copyright = f"{year}, Kyle Bystrom"
# The short X.Y version.
v,sv = __version__.split('.')[:2]
version = "%s.%s"%(v,sv)
# The full version, including alpha/beta/rc tags.
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'ciderdocext',
    'breathe',
]
bibtex_bibfiles = ['refs/refs.bib', 'refs/cider_refs.bib']

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

