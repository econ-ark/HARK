from __future__ import annotations

import importlib.metadata
import warnings
from datetime import date

try:
    import numba
except ImportError:
    pass
else:
    warnings.filterwarnings(
        "ignore",
        message="numba.generated_jit.*",
        category=numba.NumbaDeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".* 'nopython' .*",
        category=numba.NumbaDeprecationWarning,
    )

# Project information
project = "HARK"
copyright = f"{date.today().year}, Econ-ARK Team"
author = "Econ-ARK Team"
version = release = importlib.metadata.version("HARK")

# General configuration
extensions = [
    # built-in extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # third-party extensions
    "nbsphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "NARK",
]

language = "en"

master_doc = "index"

# Synchronise with Sphinx requirement in 'requirements/dev.txt'
needs_sphinx = "6.1"

pygments_style = "sphinx"

source_suffix = [".rst", ".md"]

# HTML writer configuration
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["override-nbsphinx-gallery.css"]

html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Econ-ARK/HARK",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/econ_ark",
            "icon": "fa-brands fa-square-twitter",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/econ-ark/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
        {
            "name": "Econ-ARK",
            "url": "https://econ-ark.org",
            "icon": "_static/econ-ark-logo.png",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
}

# Point to Econ-ARK repo for edit buttons
html_context = {
    "github_url": "https://github.com",
    "github_user": "econ-ark",
    "github_repo": "HARK",
    "github_version": "master",
    "doc_path": "docs/",
}

# Use Econ-ARK URL to host the website
html_baseurl = "https://docs.econ-ark.org"

html_logo = "images/econ-ark-logo.png"
html_favicon = "images/econ-ark-logo.png"
html_domain_indices = False
html_copy_source = False

# sphinx.ext.intersphinx configuration
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# sphinx.ext.autodoc configuration
autodoc_default_flags = ["members"]  # must add outside ']' bracket

# sphinx.ext.autosummary configuration
autosummary_generate = True

# sphinx.ext.napoleon configuration
napoleon_use_ivar = True  # solves duplicate object description warning

# nbsphinx configuration
nbsphinx_execute = "never"  # notebooks are executed via ``nb_exec.py``

myst_enable_extensions = ["colon_fence"]

nitpick_ignore = [("py:class", "_io.StringIO"), ("py:class", "_io.BytesIO")]

always_document_param_types = True
