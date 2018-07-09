# Packaging HARK

WORK IN PROGRESS

Early developer releases of HARK are available for `pip install` at
http://pypi.org/project/econ-ark and for `conda install` at
https://anaconda.org/sumanah-changeset/econ-ark .

The current approach for packaging draws from these tutorials:

1. https://packaging.python.org/tutorials/packaging-projects/
2. https://conda.io/docs/user-guide/tutorials/build-pkgs-skeleton.html

You will need to make a pypi.org account and a test.pypi.org account,
and go through email verification for both sites, and make an
anaconda.org account. Let @brainwane or @jasonaowen know when you have
done so, so they can give you credentials for the relevant projects on
Test PyPI, PyPI, and Anaconda.org.


## Ensure you're working on master

Switch to the `master` branch of HARK:

> git checkout master

Update your local checkout of the repository, e.g.,

> git pull origin master

## Make source distribution

Create a Python 2.7 virtualenv. Ensure that you have `twine`, `setuptools`, and `pip` installed.

Update the version number in `setup.py`.

Update the date and the version number in `README.md`.

Make a git commit with the changes you've just made.

Follow the directions within https://packaging.python.org/tutorials/packaging-projects/ to make a source distribution: `python setup.py sdist`

Deactivate the virtualenv, and make or use a different Python 2.7 virtualenv to test that you can `pip install` the `sdist` you've just made. One thing you can do to quickly verify that the package installs OK:

`>>> import HARK.simulation`
`>>> HARK.simulation.drawBernoulli(5)`

You should get something like:

`array([False, False, False, False,  True])`

## Upload to PyPI

Switch back to the virtualenv that has `twine` installed.

Follow the directions at https://packaging.python.org/guides/using-testpypi/ to upload the sdist you've just made to Test PyPI.

Switch to a fresh Python 2.7 virtualenv. Run: `pip install  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple econ-ark==VERSIONNUMBER` (substituting the new version number in for VERSIONNUMBER) and test that it installs correctly.

Go to http://test.pypi.org/project/econ-ark and look in the release history for the release you just made. Verify that the README renders fine and that the date and version number are correct.

Once you've verified the README renders correctly and the package installs correctly, run:

`twine upload SDIST.TAR.GZ` (substituting the package filename for SDIST.TAR.GZ).

## Make conda packages

Install Anaconda per the directions in the main HARK README.

Install `conda-build` per the instructions at https://conda.io/docs/user-guide/tasks/build-packages/install-conda-build.html .

Switch into your home directory and deactivate your virtualenv.

Follow the directions in https://conda.io/docs/user-guide/tutorials/build-pkgs-skeleton.html to make a `conda` package.

Follow [the "converting conda package for other platforms" instructions](https://conda.io/docs/user-guide/tutorials/build-pkgs-skeleton.html#optional-converting-conda-package-for-other-platforms) to make `conda` packages for all platforms.

## Upload to Anaconda.org

Follow [the "Uploading packages to Anaconda.org" instructions](https://conda.io/docs/user-guide/tutorials/build-pkgs-skeleton.html#optional-uploading-packages-to-anaconda-org) for all the `conda` packages you made.

Go to your anaconda.org dashboard. Verify that the packages uploaded.

Go to https://anaconda.org/sumanah-changeset/econ-ark and work with @brainwane or @jasonaowen to get your new packages onto the currently recommended channel.
