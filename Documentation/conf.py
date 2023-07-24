import warnings

try:
    import numba
except ImportError:
    pass
else:
    warnings.filterwarnings("ignore",
                            message="numba.generated_jit.*",
                            category=numba.NumbaDeprecationWarning)
    warnings.filterwarnings("ignore",
                            message=".* 'nopython' .*",
                            category=numba.NumbaDeprecationWarning)

# Project information
project = "HARK"
copyright = "2020, Econ-ARK team"
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

html_theme_options = {
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
    ]
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

# sphinx.ext.napoleon configuration
napoleon_use_ivar = True  # solves duplicate object description warning

# nbsphinx configuration
nbsphinx_execute = "never"  # This is currently not working
