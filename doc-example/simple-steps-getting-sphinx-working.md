Basic steps to get Sphinx running:

1. Download and install Sphinx, numpydoc:
    $ conda update conda
    $ conda install sphinx
    $ conda install numpydoc

2. create "doc" folder in your code directory, navigate there. For purpose of illustration will assume project is in ~/workspace/HARK
    $ cd ~/workspace/HARK
    $ mkdir doc
    $ cd doc

3. Run Sphinx quickstart and select "autodoc" and others you would like:
    $ sphinx-quickstart
   I choose these:
    > autodoc: automatically insert docstrings from modules (y/n) [n]: y
    > doctest: automatically test code snippets in doctest blocks (y/n) [n]: y
    > intersphinx: link between Sphinx documentation of different projects (y/n) [n]: y
    > todo: write "todo" entries that can be shown or hidden on build (y/n) [n]: y
    > coverage: checks for documentation coverage (y/n) [n]: y
    > imgmath: include math, rendered as PNG or SVG images (y/n) [n]: n
    > mathjax: include math, rendered in the browser by MathJax (y/n) [n]: y
    > ifconfig: conditional inclusion of content based on config values (y/n) [n]: n
    > viewcode: include links to the source code of documented Python objects (y/n) [n]: n
    > githubpages: create .nojekyll file to publish the document on GitHub pages (y/n) [n]: n

4. Open conf.py file with favorite text editor:

    $ cd ~/workspace/HARK/doc   # if not already here
    $ atom conf.py &

5. Changes to make:
    - find this line and add ".." instead of ".":

        sys.path.insert(0, os.path.abspath('..'))

    - ensure autosummary and numpydoc are included and add two lines for autodoc and autosummary below:

        extensions = [ ...             # Leave whatever options were auto-included
            'sphinx.ext.autosummary',
            'numpydoc',
            ]
        autodoc_default_flags = ['members']  # must add outside ']' bracket
        autosummary_generate = True          

    - Change theme to your favorite; more here: /home/npalmer/workspace/HARK-DOCS/HARK-docs-versions/doc-v2.0/conf.py
        html_theme = 'classic' # 'alabaster' is default
        # Others: sphinx_rtd_theme, sphinxdoc, scrolls, agogo, traditional, nature, haiku, pyramid, bizstyle

6. use sphinx-apidoc to create the .rst files for each module to document -- otherwise must write each by hand, which is what we are avoiding by using Sphinx:
    - use sphinx-apidoc:
        $ cd ~/workspace/HARK/doc   # if not already here
        $ sphinx-apidoc -f -o ./ ../
    - NOTE: syntax is:
        * '-f' force overwrite of html files
        * '-o' Required: where to find the *source .rst files for sphinx*; we are using 'doc' directly
        * './' target for '-o'
        * '../' where to look for the Python files to pull code out of
    - NOTE that when we want to create these for other files, such as ConsumptionSavingModel, we will need to indicate that here as well. Run this to generate .rst files for ConsumptionSavingModel:
        $ cd ~/workspace/HARK/doc   # if not already here
        $ sphinx-apidoc -f -o ./ ../ConsumptionSavingModel/

7. Edit the main "index.rst" file to tell it explicitly what modules to include:
    $ cd ~/workspace/HARK/doc   # if not already here
    $ atom index.rst &

8. Insert the following "autosummary" text between 'Contents' and 'Indices and tables'
   **Very important note:** .rst files use indentions to designate collections; the modules listed under ".. autosummary::" (such as HARKutilities and HARKsimulation) **must** line up with the first colon ":" before ":toctree: generated"



**EXAMPLE:**

--------------------------------------------------------------------------------

Welcome to HARK's documentation!
================================

Contents:

.. toctree::
   :maxdepth: 2

********************* START NEW ********************** [Delete this line]
.. autosummary::
  :toctree: generated

  HARKutilities
  HARKsimulation
  HARKparallel
  HARKinterpolation
  HARKestimation
  HARKcore

*********************** END NEW ********************** [Delete this line]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

--------------------------------------------------------------------------------

9. The index.rst module will now automatically generate summaries of each of the indicated modules. *NOTE* that if we want to include others that are inside other folders, we will need to indicate the path as something like 'ConsumptionSavingModel.ConsumptionSavingModel'

**Before** adding ConsumptionSavingModel.ConsumptionSavingModel , we will need to make sure that any hardcoded pathnames in the code are replaced with appropriate programatically determined pathnames. **TODO** -- still in progress


10. Add a Very Brief Welcome Text -- **TODO** -- this can be raw text directly above "Contents" or something as included in a seperate .rst file we reference before "Contents"

10.5. NOTE: If you do not have joblib installed, Sphinx will fail when it attempts to run HARKParallel. If you do not want to install joblib, remove HARKParallel from the index.rst file. To install: conda install joblib should install joblib to your anaconda package.


11. Run

    $ cd ~/workspace/HARK/doc   # if not already here
    $ make html

12. You'll get a billion warnings, I think mostly because some things are missing documentations. Regardless,
    open this and observe the nice-looking API/docs. Be sure to try the search and index features!

    $ ~/workspace/HARK/doc/_build/html/index.html


14. Update: creating of these docs for the website was accomplished following this tutorial: https://daler.github.io/sphinxdoc-test/includeme.html  This approach is nice because it allows one to maintain the code physically in one location (simplifying creation of the docs) and the html output in another location. When all is done in the same physical directory, there is extensive switching between branches to accomplish the docs update. 
Important steps include:
    - in Makefile, appropriately changing the relative path to BUILDDIR
        - NOTE: this may be particularly important for changing the "windows make file" as well, however I do not have a windows machine to test this on. 
    - Note: I did not use any of the "pdf manual" options. 
    - adding the .nojekyll file to the appropriate place



15. Steps to update docs and post:
    $ sphinx-apidoc -f -o ./ ../Module-name-to-document  # recall, also need to insert in index.rst
    $ make html 
    $ cd ../../HARK-docs
    $ git branch           # confirm on gh-pages branch
    $ git push origin master





_Useful references:_


- One of the authors, useful presentation: http://www.slideshare.net/shimizukawa/sphinx-autodoc-automated-api-documentation-pyconapac2015
    - https://www.youtube.com/watch?v=mdtxHjH2wog

- High-level, friendly overview (note install approach is deprecated):
    - https://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/
        - https://codeandchaos.wordpress.com/2012/08/09/sphinx-and-numpydoc/
    - http://gisellezeno.com/tutorials/sphinx-for-python-documentation.html
    - http://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html

- Tutorial:
    - http://sphinx-tutorial.readthedocs.io/
    - http://matplotlib.org/sampledoc/index.html
        - very nice for details of sphinx setup in quickly digestible, reproducible format.
        - see here for nice example of including "welcome text."
