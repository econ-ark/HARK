# Project information
project = "HARK"
copyright = "2020, Econ-ARK team"
author = "Econ-ARK team"
version = release = "latest"

# General configuration
extensions = [
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
    "nbsphinx",
    "recommonmark",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# HTML writer configuration
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Extend theme width
html_css_files = ["theme_overrides.css"]

# sphinx.ext.intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# sphinx.ext.autodoc configuration
autodoc_default_flags = ["members"]  # must add outside ']' bracket

# sphinx.ext.autosummary configuration
autosummary_generate = True

# nbsphinx configuration
nbsphinx_execute = "never"  # This is currently not working
