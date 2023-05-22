# Synchronise with Sphinx requirement in 'requirements/dev.txt'
needs_sphinx = "6.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgconverter",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "recommonmark",
]

# This is currently not working
nbsphinx_execute = "never"

# Extend theme width
html_css_files = ["theme_overrides.css"]

autodoc_default_flags = ["members"]  # must add outside ']' bracket
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = [
    ".rst",
    ".md",
]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "HARK"
copyright = "2020, Econ-ARK team"
author = "Econ-ARK team"

version = release = "latest"

language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

pygments_style = "sphinx"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
