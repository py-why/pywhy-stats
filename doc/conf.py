# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy.typing
import sphinx_gallery  # noqa: F401
from sphinx_gallery.sorting import ExampleTitleSortKey

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../"))

import pywhy_stats  # noqa: E402

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "pywhy_stats")))

# -- Project information -----------------------------------------------------

project = "pywhy_stats"
copyright = f"{datetime.today().year}, PyWhy"
author = "PyWhy"
version = pywhy_stats.__version__

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "5.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_issues",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx.ext.graphviz",
]

graphviz_output_format = "png"

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# generate autosummary even if no references
# -- sphinx.ext.autosummary
autosummary_generate = True

autodoc_default_options = {"inherited-members": None}

# whether to expand type hints in function/class signatures
autodoc_typehints = "none"

add_module_names = False

# -- numpydoc
# Below is needed to prevent errors
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True
numpydoc_validate = True

numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    "self.verbose",
    "py",
    "the",
    "functions",
    "lambda",
    "container",
    "iterator",
    "keyword",
    "arguments",
    "dictionary",
    "no",
    "attributes",
    # shapes
    "n_times",
    "obj",
    "arrays",
    "lists",
    "func",
    "n_nodes",
    "n_estimated_nodes",
    "n_samples",
    "n_variables",
    "n_classes",
    "n_samples_X",
    "n_samples_Y",
    "n_features_x",
    "n_features_y",
    "n_features",
    "n_features_z",
    "k",
    "n_features_X",
    "n_features_Y",
    # deep learning
    "pytorch",
    "model",
}
numpydoc_xref_aliases = {
    "pgmpy.models.BayesianNetwork": "pgmpy.models.BayesianNetwork",
    # joblib
    "joblib.Parallel": "joblib.Parallel",
    "PValueResult": "pywhy_stats.pvalue_result.PValueResult",
    # numpy
    "NDArray": "numpy.ndarray",
    "ArrayLike": ":term:`array_like`",
}

autodoc_typehints_format = "short"
# from __future__ import annotations
# autodoc_type_aliases = {
#     'Iterable': 'Iterable',
#     'ArrayLike': 'ArrayLike'
# }
default_role = "literal"

# Tell myst-parser to assign header anchors for h1-h3.
# myst_heading_anchors = 3
# suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "auto_examples/index.rst",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "auto_examples/*.rst",
]

source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "pgmpy": ("https://pgmpy.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
intersphinx_timeout = 5

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Clean up sidebar: Do not show "Source" link
# html_show_sourcelink = False
# html_copy_source = False

html_theme = "pydata_sphinx_theme"

html_title = f"pywhy-stats v{version}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/pywhy/pywhy-stats",
            icon="fab fa-github-square",
        ),
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}

scrapers = ("matplotlib",)
# Add pygraphviz png scraper, if available
try:
    from pygraphviz.scraper import PNGScraper

    scrapers += (PNGScraper(),)
except ImportError:
    pass

sphinx_gallery_conf = {
    "doc_module": "pywhy_stats",
    "reference_url": {
        "pywhy_stats": None,
    },
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "within_subsection_order": ExampleTitleSortKey,
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": "^((?!sgskip).)*$",
    "matplotlib_animations": True,
    "compress_images": ("images", "thumbnails"),
    "image_scrapers": scrapers,
}

# prevent jupyter notebooks from being run even if empty cell
# nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html"],
}

html_context = {
    "versions_dropdown": {
        "dev": "v0.2 (devel)",
        "stable": "v0.1 (stable)",
        "v0.1": "v0.1",
    },
}

# Enable nitpicky mode - which ensures that all references in the docs
# resolve.

nitpicky = False
nitpick_ignore = [
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    ("py:class", "numpy._typing._array_like._SupportsArray"),
    ("py:class", "numpy._typing._nested_sequence._NestedSequence"),
]
