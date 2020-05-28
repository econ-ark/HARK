Developer overview
==================

1. If you are a first-time contributor:

   * Go to `https://github.com/econ-ark/HARK
     <https://github.com/econ-ark/HARK>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer::

      git clone git@github.com:your-username/HARK.git

   * Navigate to the folder HARK and add the upstream repository::

      git remote add upstream git@github.com:econ-ark/HARK.git

   * Now, you have remote repositories named:

      - ``upstream``, which refers to the ``HARK`` repository
      - ``origin``, which refers to your personal fork of ``HARK``.

2. Develop your contribution:

   * Pull the latest changes from upstream::

      git checkout master
      git pull upstream master

   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'bugfix-for-issue-220'::

      git checkout -b bugfix-for-issue-220

   * Commit locally as you progress (``git add`` and ``git commit``)

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin bugfix-for-issue-220

   * Go to GitHub. The new branch will show up with a green Pull Request
     button---click it.


4. Review process:

    * Reviewers (the other developers and interested community members) will
      write inline and/or general comments on your Pull Request (PR) to help
      you improve its implementation, documentation, and style.  Every single
      developer working on the project has their code reviewed, and we've come
      to see it as friendly conversation from which we all learn and the
      overall code quality benefits.  Therefore, please don't let the review
      discourage you from contributing: its only aim is to improve the quality
      of project, not to criticize (we are, after all, very grateful for the
      time you're donating!).

    * To update your pull request, make your changes on your local repository
      and commit. As soon as those changes are pushed up (to the same branch as
      before) the pull request will update automatically.

    * `Travis-CI <https://travis-ci.org/>`_, a continuous integration service,
      is triggered after each Pull Request update to build the code and run unit
      tests of your branch. The Travis tests must pass before your PR can be merged.
      If Travis fails, you can find out why by clicking on the "failed" icon (red
      cross) and inspecting the build and test log.

    * `GitHub Actions <http://github.com>`_, is another continuous integration
      service, which we use.  You will also need to make sure that the GitHub Actions
      tests pass.

.. note::

   If closing a bug, also add "Fixes #1480" where 1480 is the issue number.

Divergence between ``upstream master`` and your feature branch
--------------------------------------------------------------

Never merge the main branch into yours. If GitHub indicates that the
branch of your Pull Request can no longer be merged automatically, rebase
onto master::

   git checkout master
   git pull upstream master
   git checkout bugfix-for-issue-1480
   git rebase master

If any conflicts occur, fix the according files and continue::

   git add conflict-file1 conflict-file2
   git rebase --continue

However, you should only rebase your own branches and must generally not
rebase any branch which you collaborate on with someone else.

Finally, you must push your rebased branch::

   git push --force origin bugfix-for-issue-1480

(If you are curious, here's a further discussion on the
`dangers of rebasing <http://tinyurl.com/lll385>`_.
Also see this `LWN article <http://tinyurl.com/nqcbkj>`_.)

Build environment setup
-----------------------

Once you've cloned your fork of the HARK repository,
you should set up a Python development environment tailored for HARK.
You may choose the environment manager of your choice.
Here we provide instructions for two popular environment managers:
``venv`` (pip based) and ``conda`` (Anaconda or Miniconda).

venv
^^^^
When using ``venv``, you may find the following bash commands useful::

  # Create a virtualenv named ``econ-dev`` that lives in the directory of
  # the same name
  python -m venv econ-dev
  # Activate it
  source econ-dev/bin/activate
  # Build and install HARK from source with developer requirements
  pip install -e ".[dev]"
  # Test your installation
  pip install pytest
  pytest HARK/

conda
^^^^^

When using conda, you may find the following bash commands useful::

  # Create a conda environment named ``econ-dev``
  conda create --name econ-dev
  # Activate it
  conda activate econ-dev
  # Install minimal testing dependencies
  conda install pytest
  # Build and install HARK from source with developer requirements
  pip install -e ".[dev]"
  # Test your installation
  pytest HARK/


Guidelines
----------

* All code should have tests.
* All code should be documented, to the same
  `standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_
  as NumPy and SciPy.
* All changes are reviewed.

Stylistic Guidelines
--------------------

* We use `black <https://black.readthedocs.io>`_ for styling of code::
    
    # install black
    pip install black
    # run black on the changed files
    black path_to_changed_file.py

Testing
-------

``HARK`` has a test suite that ensures correct
execution on your system.  The test suite has to pass before a pull
request can be merged, and tests should be added to cover any
modifications to the code base.

We make use of the `pytest <https://docs.pytest.org/en/latest/>`__ and unittests
testing framework, with tests located in the various
``HARK/submodule/tests`` folders.

To use ``pytest``, ensure that the library is installed in development mode::

    $ pip install -e .

Now, run all tests using::

    $ pytest HARK

Or the tests for a specific submodule::

    $ pytest HARK/ConsumptionSaving

Or tests from a specific file::

    $ pytest HARK/ConsumptionSaving/tests/test_ConsAggShockModel.py


Pull request codes
------------------

When you submit a pull request to GitHub, GitHub will ask you for a summary.  If
your code is not ready to merge, but you want to get feedback, please consider
using ``WIP: experimental optimization`` or similar for the title of your pull
request. That way we will all know that it's not yet ready to merge and that
you may be interested in more fundamental comments about design.

When you think the pull request is ready to merge, change the title (using the
*Edit* button) to remove the ``WIP:``.


Bugs
----

Please `report bugs on GitHub <https://github.com/econ-ark/HARK/issues>`_.
