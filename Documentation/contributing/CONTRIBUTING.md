# Contributing to Econ-ARK

* [Code of Conduct](#code-of-conduct)
* [Contributing Guide](#contributing-guide)
* [Developer's Certificate of Origin 1.1](#developers-certificate-of-origin-11)

## [Code of Conduct](./doc/guides/contributing/coc.md)

The Econ-ARK project has a
[Code of Conduct](./.github/CODE_OF_CONDUCT.md)
to which all contributors must adhere.

See [details on our policy on Code of Conduct](./.github/CODE_OF_CONDUCT.md).


## Welcome!

Thank you for considering contributing to Econ-ARK!  We're a young project with a small but committed community that's hoping to grow while maintaining our friendly and responsive culture.  Whether you're an economist or a technologist, a writer or a coder, an undergrad or a full professor, a professional or a hobbyist, there's a place for you in the Econ-ARK community. 

We're still creating our contribution infrastructure, so this document is a work in progress.  If you have any questions please feel free to @ or otherwise reach out project manager [Shauna](https://github.com/shaunagm), or lead developers [Chris](https://github.com/llorracc) and [Matt](https://github.com/mnwhite).  If you prefer to connect through email, you can send it to __econ-ark at jhuecon dot org__.

## How to Contribute

We're open to all kinds of contributions, from bug reports to help with our docs to suggestions on how to improve our code.  The best way to figure out if the contribution you'd like to make is something we'd merge or otherwise accept, is to open up an issue in our issue tracker.  Please create an issue rather than immediately submitting pull request, unless the change you'd like to make is so minor you won't mind if the pull request is rejected.  For bigger contributions, we want to proactively talk things through so we don't end up wasting your time.

While we're thrilled to receive all kinds of contributions, there are a few key areas we'd especially like help with:

* porting existing heterogenous agent/agent based models into HARK
* curating and expanding the collection of projects which use Econ-ARK (which we store in the [remark](https://github.com/econ-ark/REMARK) repository)
* creating demonstrations of how to use Econ-ARK (which we store in the [DemARK](https://github.com/econ-ark/DemARK) repository)
* expanding test coverage of our existing code

If you'd like to help with those or any other kind of contribution, reach out to us and we'll help you do so.  

We don't currently have guidelines for opening issues or pull requests, so include as much information as seems relevant to you, and we'll ask you if we need to know more.

## Responding to Issues & Pull Requests

We're trying to get better at managing our open issues and pull requests.  We've created a new set of goals for all issues and pull requests in our Econ-ARK repos:

1. Initial response within one or two days.
2. Substantive response within two weeks.
3. Resolution of issue/pull request within three months.  

If you've been waiting on us for more than two weeks for any reason, please feel free to give us a nudge.  Correspondingly, we ask that you respond to any questions or requests from us within two weeks as well, even if it's just to say, "Sorry, I can't get to this for a while yet". If we don't hear back from you, we may close your issue or pull request.  If you want to re-open it, just ask - we're glad to do so.

## Getting Started

The Contributing Guide below provides instructions for how to get started running HARK.  This also serves as a setup guide for new contributors.  If you run into any problems, please let us know by opening an issue in the issue tracker.

Thanks again! We're so glad to have you in our community.

### Contributing Guide

1. If you are a first-time contributor:

   * Go to [https://github.com/econ-ark/HARK](https://github.com/econ-ark/HARK) and click the
     "fork" button to create your own copy of the project.

   * Clone the project to your local computer
    ```
      git clone git@github.com:your-username/HARK.git
    ```
   * Navigate to the folder HARK and add the upstream repository
    ```
      git remote add upstream git@github.com:econ-ark/HARK.git
    ```
   * Now, you have remote repositories named:

      - ``upstream``, which refers to the ``HARK`` repository
      - ``origin``, which refers to your personal fork of ``HARK``.

2. Develop your contribution:

   * Pull the latest changes from upstream
    ```
      git checkout master
      git pull upstream master
    ```
   * Create a branch for the feature you want to work on. Since the
     branch name will appear in the merge message, use a sensible name
     such as 'bugfix-for-issue-220'
    ```
      git checkout -b bugfix-for-issue-220
    ```
   * Commit locally as you progress (``git add`` and ``git commit``)

3. To submit your contribution:

   * Push your changes back to your fork on GitHub
    ```
      git push origin bugfix-for-issue-220
    ```
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

    * [Travis-CI](https://travis-ci.org/), a continuous integration service,
      is triggered after each Pull Request update to build the code and run unit
      tests of your branch. The Travis tests must pass before your PR can be merged.
      If Travis fails, you can find out why by clicking on the "failed" icon (red
      cross) and inspecting the build and test log.

    * [GitHub Actions](http://github.com), is another continuous integration
      service, which we use.  You will also need to make sure that the GitHub Actions
      tests pass.

NOTE:  If closing a bug, also add "Fixes #1480" where 1480 is the issue number.


### Build environment setup

Once you've cloned your fork of the HARK repository,
you should set up a Python development environment tailored for HARK.
You may choose the environment manager of your choice.
Here we provide instructions for two popular environment managers:
``venv`` (pip based) and ``conda`` (Anaconda or Miniconda).

#### venv

When using ``venv``, you may find the following bash commands useful
```
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
```

#### conda

When using conda, you may find the following bash commands useful
```
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
```

### Guidelines

* All code should have tests.
* All code should be documented, to the same
  [standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard)
  as NumPy and SciPy.
* All changes are reviewed.

### Stylistic Guidelines

* We use `black <https://black.readthedocs.io>`_ for styling of code
```
    # install black
    pip install black
    # run black on the changed files
    black path_to_changed_file.py
```

### Testing

``HARK`` has a test suite that ensures correct
execution on your system.  The test suite has to pass before a pull
request can be merged, and tests should be added to cover any
modifications to the code base.

We make use of the [pytest](https://docs.pytest.org/en/latest/) and unittests
testing framework, with tests located in the various
``HARK/submodule/tests`` folders.

To use ``pytest``, ensure that the library is installed in development mode
```
    $ pip install -e .
```
Now, run all tests using
```
    $ pytest HARK
```
Or the tests for a specific submodule
```
    $ pytest HARK/ConsumptionSaving
```
Or tests from a specific file
```
    $ pytest HARK/ConsumptionSaving/tests/test_ConsAggShockModel.py
```

### Pull request codes

When you submit a pull request to GitHub, GitHub will ask you for a summary.  If
your code is not ready to merge, but you want to get feedback, please consider
using ``WIP: experimental optimization`` or similar for the title of your pull
request. That way we will all know that it's not yet ready to merge and that
you may be interested in more fundamental comments about design.

When you think the pull request is ready to merge, change the title (using the
*Edit* button) to remove the ``WIP:``.


### Bugs

Please [report bugs on GitHub](https://github.com/econ-ark/HARK/issues).


## Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

* (a) The contribution was created in whole or in part by me and I
  have the right to submit it under the open source license
  indicated in the file; or

* (b) The contribution is based upon previous work that, to the best
  of my knowledge, is covered under an appropriate open source
  license and I have the right under that license to submit that
  work with modifications, whether created in whole or in part
  by me, under the same open source license (unless I am
  permitted to submit under a different license), as indicated
  in the file; or

* (c) The contribution was provided directly to me by some other
  person who certified (a), (b) or (c) and I have not modified
  it.

* (d) I understand and agree that this project and the contribution
  are public and that a record of the contribution (including all
  personal information I submit with it, including my sign-off) is
  maintained indefinitely and may be redistributed consistent with
  this project or the open source license(s) involved.
