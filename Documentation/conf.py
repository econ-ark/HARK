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
        "ignore", message=".* 'nopython' .*", category=numba.NumbaDeprecationWarning
    )

# Project information
project = "HARK"
copyright = f"{date.today().year}, Econ-ARK team"
author = "Econ-ARK team"
version = release = "latest"

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
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "NARK",
]

language = "en"

master_doc = "index"

# Synchronise with Sphinx requirement in 'requirements/dev.txt'
needs_sphinx = "6.1"

pygments_style = "sphinx"

source_suffix = [
    ".rst",
    ".md",
]

# HTML writer configuration
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "override-nbsphinx-gallery.css",
]

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
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "index": ["page-toc"],
    },
}

nbsphinx_prolog = r"""
{% set docname = 'Documentation/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/econ-ark/HARK/tree/master/{{ docname|e }}">{{ docname|e }}</a>.
      <br />
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/econ-ark/HARK/master?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download>Download notebook</a>.
    </div>
"""
myst_enable_extensions = [
    "colon_fence",
]
# Point to Econ-ARK repo for edit buttons
html_context = {
    "github_url": "https://github.com",
    "github_user": "econ-ark",
    "github_repo": "hark",
    "github_version": "master",
    "doc_path": "Documentation/",
}

# Use Econ-ARK URL to host the website
html_baseurl = "https://docs.econ-ark.org"

html_logo = "images/econ-ark-logo.png"
html_favicon = "images/econ-ark-logo.png"
html_domain_indices = False
html_copy_source = False

# sphinx.ext.intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# sphinx.ext.autodoc configuration
autodoc_default_flags = ["members"]  # must add outside ']' bracket

# sphinx.ext.autosummary configuration
autosummary_generate = True

# Orders functions by the source order
autodoc_member_order = "bysource"

# sphinx.ext.napoleon configuration
napoleon_use_ivar = True  # solves duplicate object description warning

# nbsphinx configuration
nbsphinx_execute = "never"  # notebooks are executed via ``nb_exec.py``
