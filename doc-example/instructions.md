Here are the steps I took to set up and run a Sphinx instance. Tutorials and instruction sets I personally found very useful are cataloged at the end of this instruction set.

You will complete the following steps:

- Install Sphinx
- Set up Sphinx for your particular project:
    - in your project, use "quickstart" to create the required Sphinx documentation infrastructure
    - edit the project's Sphinx configuration file ("conf.py") to ensure preferred options are used
    - edit the project's main index file ("index.rst") to direct Sphinx to document (or auto-document) the correct sections of your project
- Edit code files to ensure that Sphinx runs correctly:
    - *because Sphinx runs all files to extract docs*:
        - ensure that appropriate "script" code calls are wrapped in "if __name__ == "__main__" blocks
        - ensure that any hardcoded filepaths instead use appropriate sys/os calls to set up pathnames
    - confirm that the appropriate document string structure is in all files that
- Run Sphinx and examine output


We'll discuss each of these steps in turn below.


## Install Sphinx

This is wonderfully simple if you have anaconda:

    $ conda update conda
    $ conda install sphinx
    $ conda install numpydoc

This should install the most recent versions of these tools. We will use
numpydoc to make nice looking documentation.

## Set up Sphinx for your particular project

The first step is running a "quickstart" program which will set up the sphinx
infrastructure for your project. Convention seems to be to use a "doc" directory,
which quickstart will create for you. (If you already have a "doc" directory,
simply create a directory with a different name; name isn't important.)

    $ cd ~/workspace/HARK/
    $ sphinx-quickstart doc

This will create a "doc" directory there and launch the quick-start command-line
interface. (Use "sphinx-doc" or some variation on "doc" if you already have a
"doc" directory that you want to use for other things.)

You will be ginve a lot of options, here are the ones I use to set up my Sphinx;
empty spots after the colon on each line indicate [default choice] selected:

    > Separate source and build directories (y/n) [n]:
    > Name prefix for templates and static dir [_ ]:
    > Project name: HARK
    > Author name(s): Christopher D. Carroll, Alexander Kaufman, David C. Low, Nathan M. Palmer, Matthew N. White
    > Project version: 0.9
    > Project release [0.9]:
    > Project language [en]:
    > Source file suffix [.rst]:
    > Name of your master document (without suffix) [index]:
    > Do you want to use the epub builder (y/n) [n]:
    > autodoc: automatically insert docstrings from modules (y/n) [n]: y
    > doctest: automatically test code snippets in doctest blocks (y/n) [n]: y
    > intersphinx: link between Sphinx documentation of different projects (y/n) [n]: y
    > todo: write "todo" entries that can be shown or hidden on build (y/n) [n]: y
    > coverage: checks for documentation coverage (y/n) [n]: y
    > imgmath: include math, rendered as PNG or SVG images (y/n) [n]: n
    > mathjax: include math, rendered in the browser by MathJax (y/n) [n]: y
    > ifconfig: conditional inclusion of content based on config values (y/n) [n]:
    > viewcode: include links to the source code of documented Python objects (y/n) [n]: n
    > githubpages: create .nojekyll file to publish the document on GitHub pages (y/n) [n]: n
    > Create Makefile? (y/n) [y]: y
    > Create Windows command file? (y/n) [y]: y

These options are used by quickstart to create the files and directories under

    ~/workspace/HARK/doc/

which will run Sphinx. If you navigate to the above directory you should see:

    _ templates/
    _ build/
    _ static/
    index.rst
    conf.py
    make.bat
    Makefile

The first three are directories which will contain output of running
autodocumentation. Eventually you will look in _ build/html/index.html to find
the "root" of the html documentation after we run the autodocs. This index.html
will be intimately connected to the "index.rst" file as described below.

The index.rst and conf.py files are where we control the setup for the output.

- conf.py:
    - controls how Sphinx will run -- this is the configuration file for Sphinx and has largely been populated by the quickstart options we selected above. We'll add a couple things in a moment.
- index.rst:
    - this controls how Sphinx will arrange the "root" of the html documentation. Essentially we are building the "table of contents" here (we will actually explicitly include a ToC command in here).
    - fun fact: if you make multiple "index.rst" files, Sphinx will dutifully create a matching "index.html" files for each ".rst" file. For example if you create three index files titled "index.rst," "index-manual-toc.rst," and "index-auto-toc.rst," after running sphinx you will get three matching index.html files under _ build/html called: "index.html," "index-manual-toc.html," and "index-auto-toc.html."
    - we can use this to try different types of index file.

Now let's edit these two files.

### Edit conf.py

Here are useful elements to add to the conf.py file. I include the previous line in the conf.py default file so you can readily find these in the file itself:

- Add a direct call to "abspath" at the beginning, *but* note that we need to add ".." instead of "." as suggested in the docs, because this file is one below the root of the code we want to document:

    # If extensions (or modules to document with autodoc) are in another directory,
    # add these directories to sys.path here. If the directory is relative to the
    # documentation root, use os.path.abspath to make it absolute, like shown here.
    #sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.abspath('..'))   # <- Careful here to add ".."

- Add numpydoc to the extensions -- most should be populated from the quickstart:



    # Add any Sphinx extension module names here, as strings. They can be
    # extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
    # ones.
    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.doctest',
        'sphinx.ext.autosummary',
        'sphinx.ext.intersphinx',
        'sphinx.ext.todo',
        'sphinx.ext.coverage',
        'sphinx.ext.mathjax',
        'numpydoc',                
    ]
    # **Be sure** to add the numpydoc file!


- Just below extensions I add these two flags to automatically document the members of modules we want to document. (See more [here](http://www.sphinx-doc.org/en/stable/ext/autodoc.html#confval-autodoc_default_flags), and [here](http://www.sphinx-doc.org/en/stable/ext/autosummary.html).)

    autodoc_default_flags = ['members']
    autosummary_generate = True          # Will generate some stubs. Can comment out to see difference.

- Finally, choose the style we want to use. I think the "classic" style works better than the default, minimal, "alabaster" style, but there are many to choose from:

    # The theme to use for HTML and HTML Help pages.  See the documentation for
    # a list of builtin themes.
    html_theme = 'classic'
    # See much more here: http://www.sphinx-doc.org/en/stable/theming.html
    # options:
    # - alabaster
    # - sphinx_rtd_theme  # read the docs theme
    # - classic
    # - sphinxdoc
    # - scrolls
    # - agogo
    # - traditional
    # - nature
    # - haiku
    # - pyramid
    # - bizstyle

That should do it for the conf.py file. Now onto the index.rst file.


### Quick-run Sphinx

Setting up conf.py is all that is needed -- we can quickly run Sphinx to see what the output looks like now before further direction.

Simply navigate to the "doc" file (should already be there but just in case you are starting new) and make html:

    $ cd ~/workspace/HARK/doc/
    $ make html

This will run Sphinx. You can find the very minimal output in "~/workspace/HARK/doc/_build/html/index.hml"
_Note:_ There will be brely anything in this file -- now time to populate it by editing index.rst.


### Run Sphinx

Just as before we run SPhinx again. If the above two "Import

Now we get to business. There are a large number of ways to build this file out.

We will use a simple one. Before edit, file contents should look like:

    # [some filler at beginning]

    Welcome to HARK's documentation!
    ================================

    Contents:

    .. toctree::
       :maxdepth: 2



    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`


We will add the following -- see between the "***... START NEW ...****" and "***... END NEW ...****" lines. Note: don't include those "START NEW" and "END NEW" lines:



    Welcome to HARK's documentation!
    ================================

    Contents:

    .. toctree::
       :maxdepth: 2


    ********************* START NEW ********************** [Delete this line]

    .. autosummary::
     :toctree: generated

     ConsumptionSavingModel.ConsumptionSavingModel
     ConsumptionSavingModel.SolvingMicroDSOPs
     ConsumptionSavingModel.ConsPrefShockModel

    *********************** END NEW ********************** [Delete this line]


    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`


This will tell Sphinx to go up one level, find the "ConsumptionSavingModel" directory, and automatically generate documentation for the modules (code files) ConsumptionSavingModel, SolvingMicroDSOPs, and ConsPrefShockModel. The ":members:" directive tells Sphinx to search all member functions in those modules (code files) for documentation.

**Important code note:** Sphinx will run code that it documents. This has two important implications:

- any code that you *don't* want to run automatically in those modules must be wrapped in an "if __name__ == '__main__'" statement.
- if there are any local pathnames referenced in those files, you will need to use the "os.path.abspath()" function from the os file to find the correct path. This is particularly important with reading in data for estimation.


### Example of Code Docs


Here is a very simple code example, taken from [this  tutorial](https://codeandchaos.wordpress.com/2012/08/09/sphinx-and-numpydoc/), for a basic "foo" example:


    def foo(var1, var2, long_var_name='hi')
        """This function does something.

        Parameters
        ----------
        var1 : array_like
            This is a type.
        var2 : int
            This is another var.
        Long_variable_name : {'hi', 'ho'}, optional
            Choices in brackets, default first when optional.

        Returns
        -------
        describe : type
            Explanation
        """
        print var1, var2, long_var_name


As the tutorial notes, this will produce docs that look like Numpy Docs (an [example](http://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.min_scalar_type.html#numpy.min_scalar_type) here).

This is good, because the default Sphinx documentation style is pretty unpleasant to look at.



### Run Sphinx

Just as before we run SPhinx again. If the above two "Important Code Notes" (about "if name==main" and "os.path.abspath()") are not a problem, this should run fine:


$ cd ~/workspace/HARK/doc/
$ make html

This will run Sphinx. You can find the very minimal output in "~/workspace/HARK/doc/_build/html/index.hml"
_Note:_ There will be barely anything in this file -- now time to populate it by editing index.rst.



# Useful links

Some extremely useful sources:

- One of the authors, useful presentation: http://www.slideshare.net/shimizukawa/sphinx-autodoc-automated-api-documentation-pyconapac2015
    - https://www.youtube.com/watch?v=mdtxHjH2wog

- High-level, friendly overview (note install approach is deprecated):
    - https://codeandchaos.wordpress.com/2012/07/30/sphinx-autodoc-tutorial-for-dummies/
    - https://codeandchaos.wordpress.com/2012/08/09/sphinx-and-numpydoc/

- Tutorial:
    - http://sphinx-tutorial.readthedocs.io/
