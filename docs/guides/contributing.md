# Contributing to Econ-ARK

## Code of Conduct

The Econ-ARK project has a [Code of Conduct][Conduct]
to which all contributors must adhere.
See details in the [written policy statement][Conduct].

[Conduct]: https://github.com/econ-ark/HARK/blob/master/.github/CODE_OF_CONDUCT.md

## Welcome!

Thank you for considering contributing to Econ-ARK!
We're a young project with a small but committed community
that's hoping to grow while maintaining our friendly and responsive culture.
Whether you're an economist or a technologist,
a writer or a coder, an undergrad or a full professor,
a professional or a hobbyist, there's a place for you in the Econ-ARK community.

We're still creating our contribution infrastructure,
so this document is a work in progress.
If you have any questions, please feel free to @
or otherwise reach out to project leaders [Chris] and [Matt].
If you prefer to connect through email,
you can send it to **econ-ark at jhuecon dot org**.
We also have a (rarely used) [Discord channel][Discord].

[Chris]: https://github.com/llorracc
[Matt]: https://github.com/mnwhite
[Gitter]: https://gitter.im/econ-ark/community
[Discord]: https://discord.gg/RwHg7sZrPY

## How to Contribute

We're open to all kinds of contributions, from bug reports to help with our
docs to suggestions on how to improve our code. The best way to figure out
if the contribution you'd like to make is something we'd merge or otherwise
accept, is to open up an issue in our issue tracker. Please create an issue
rather than immediately submitting pull request, unless the change you'd like
to make is so minor you won't mind if the pull request is rejected. For
bigger contributions, we want to proactively talk things through, so that we
don't end up wasting your time.

While we're thrilled to receive all kinds of contributions,
there are a few key areas we'd especially like help with:

- porting existing heterogeneous agent/agent-based models into HARK
- curating and expanding the collection of projects which use Econ-ARK
  (which we store in the [REMARK] repository)
- creating demonstrations of how to use Econ-ARK
  (which we store in the [DemARK] repository)
- expanding test coverage of our existing code

[REMARK]: https://github.com/econ-ark/REMARK
[DemARK]: https://github.com/econ-ark/DemARK

If you'd like to help with those or any other kind of contribution,
reach out to us, and we'll help you to do so.

We don't currently have guidelines for opening issues or pull requests,
so include as much information as seems relevant to you,
and we'll ask you if we need to know more.

## Responding to Issues and Pull Requests

We're trying to get better at managing our open issues and pull requests.
We've created a new set of goals for all issues and pull requests in our Econ-ARK repos:

1. Initial response within one or two days.
2. Substantive response within two weeks.
3. Resolution of issue/pull request within three months.

If you've been waiting on us for more than two weeks for any reason,
please feel free to give us a nudge.
Correspondingly, we ask that you respond to any questions or requests
from us within two weeks as well,
even if it's just to say, "Sorry, I can't get to this for a while yet".
If we don't hear back from you, we may close your issue or pull request.
If you want to re-open it, just ask---we're glad to do so.

## Getting Started

The Contributing Guide below provides instructions for how to get started running HARK.
This also serves as a setup guide for new contributors.
If you run into any problems, please let us know by opening an issue in the issue tracker.

Thanks again! We're so glad to have you in our community.

### Contributing Guide

1. If you are a first-time contributor:

   - Go to https://github.com/econ-ark/HARK and click the
     "fork" button to create your own copy of the project. If you are new to GitHub, you can perform the next steps using GitHub Desktop as well.
   - Clone the project to your local computer

     ```bash
     git clone git@github.com:your-username/HARK
     ```

   - Navigate to the folder HARK and add the upstream repository

     ```bash
     git remote add upstream git@github.com:econ-ark/HARK
     ```

   - Now, have remote repositories named:

     - `upstream`, which refers to the `HARK` repository
     - `origin`, which refers to your personal fork of `HARK`.

2. Develop your contribution:

   - Pull the latest changes from upstream

     ```
     git checkout master
     git pull upstream master
     ```

   - Create a branch for the feature you want to work on.
     Since the
     branch name will appear in the merge message, use a sensible name
     such as 'bugfix-for-issue-220'

     ```
     git checkout -b bugfix-for-issue-220
     ```

   - Commit locally as you progress (`git add` and `git commit`)

3. To submit your contribution:

   - Push your changes back to your fork on GitHub

     ```
     git push origin bugfix-for-issue-220
     ```

   - Go to GitHub.
     The new branch will show up with a green Pull Request
     button---click it.

4. Review process:

   - Reviewers (the other developers and interested community members) will
     write inline and/or general comments on your Pull Request (PR) to help
     you improve its implementation, documentation, and style.
     Every single
     developer working on the project has their code reviewed, and we've come
     to see it as friendly conversation from which we all learn and the
     overall code quality benefits.
     Therefore, please don't let the review
     discourage you from contributing: its only aim is to improve the quality
     of the project, not to criticize
     (we are, after all, very grateful for the time you're donating!).

   - To update your pull request, make your changes in your local repository
     and commit.
     As soon as those changes are pushed up
     (to the same branch as before)
     the pull request will update automatically.

   - [GitHub Actions](https://github.com/econ-ark/HARK/actions),
     a continuous integration service,
     is triggered after each Pull Request update to build the code and run unit
     tests of your branch.
     The tests must pass before your PR can be merged.
     If the tests fail, you can find out why by clicking on the "failed" icon
     (red cross) and inspecting the build and test log.

NOTE: If closing a bug, also add "Fixes #1480" where 1480 is the issue number.

### Build environment setup

Once you've cloned your fork of the HARK repository,
you should set up a Python development environment tailored for HARK.
HARK currently supports Python 3.10 to 3.13; we recommend version 3.13.
You may choose the environment manager of your choice.
Here we provide instructions for two popular environment managers:
`venv` (pip-based) and `conda` (Anaconda or Miniconda).

#### venv

When using `venv`, you may find the following bash commands useful

```bash
# Create a virtualenv named ``econ-dev`` that lives in the directory of
# the same name
python -m venv econ-dev
# Activate it
source econ-dev/bin/activate
# Build and install HARK from source with developer requirements
pip install -e ".[dev]"
# Test your installation
pytest HARK/
```

#### conda

When using conda, you may find the following bash commands useful

```bash
# Create a conda environment named ``econ-dev``
conda create -n econ-dev python=3.13
# Activate it
conda activate econ-dev
# Build and install HARK from source with developer requirements
pip install -e ".[dev]"
# Test your installation
pytest HARK/
```

### Guidelines

- All code should have tests.
- All code should be documented, to the same
  [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
  as NumPy and SciPy.
- All changes are reviewed.
- [`ruff`][Ruff] is used to handle code style requirements.

[Ruff]: https://docs.astral.sh/ruff/

### Naming Conventions

The following conventions apply throughout HARK:

- Functions and methods are always in ''snake case'': underscores separating
  each word in the name; capital letters only for abbreviations (e.g. `SS`
  for "steady state") or when referring to a variable name.

- Function and method names should accurately and concisely describe what the function does;
  the first word in the name _must be a verb_ in the imperative tense
  (e.g. `make_assets_grid`).

- Variable and class names _should not_ have a verb as their first word.

- Class names should use no underscores and capitalize the first letter of each word;
  moreover, a class name _must include a noun_, e.g. **_ConsumerSolution_**.

When naming variables in model modules, the HARK team strongly discourages
using single letter names, like `c` for consumption. Likewise, you should
generally not use variable names that are simply the math symbol for that
variable, like `theta`. The primary exception to this rule is when writing
small functions (2-5 lines) whose purpose is clear in context: the function's
name is descriptive, and when it is actually called, "properly named"
variables are passed to it.

In general, we encourage contributors to use longer, more descriptive variable
names using additional words and common abbreviations to specify the variable
more precisely. You may find it useful to look through HARK's code to get a
sense of the variable naming style, but here are some commonly used abbreviations:

- `k` : capital holdings at the very beginning of a period
- `c` : consumption
- `m` : market resources (cash on hand)
- `a` : end-of-period assets (after all actions are accomplished)
- `b` : bank balances after interest but before labor income
- `y` : labor income
- `p` : permanent/persistent income or productivity
- `h` : human wealth (usually) OR health
- `v` : value (in the Bellman sense)
- `t` : time generally; can refer to "this period" when used as `_t`
- `T` : terminal period, or the count of periods in some sense
- `Lvl` : absolute level
- `Nrm` : normalized by permanent/persistent income level
- `Now` : pertaining to the current period
- `Next` : pertaining to the succeeding period
- `Prev` : pertaining to the prior period
- `Func` : function
- `P` : "prime" as a shorthand for first derivative of a univariate function
- `PP` : "prime prime" as a shorthand for second derivative of a univariate function
- `Nvrs` : "pseudo-inverse", values decurved by the inverse (marginal) utility function
- `Xtra` : quantity above the minimum allowable for something
- `Dstn` : distribution
- `Shk` : shock or random variable
- `Cnst` : constrained or constraint
- `Unc` : un-constrained
- `Perm` : pertaining to the permanent or persistent component of income
- `Trans` : pertaining to the transitory component of income
- `Prb` : probability
- `Ex` : expected or expectation
- `Avg` : average or mean (is often `Mean`)
- `Std` : standard deviation
- `Var` : variance
- `Val` : values (as in numeric quantities)
- `Init` : initial, as in "at the moment the model starts"
- `MPC` : marginal propensity to consume
- `Min` : minimum, lower bound, or infimum
- `Max` : maximum, upper bound, or supremum
- `SS` : steady state
- `PF` : perfect foresight
- `EndOfPrd` : end-of-period
- `Fac` : factor
- `Rte` : rate
- `Liv` : live or survive
- `Boro` : borrow

These abbreviations are usually combined to make short but descriptive names.
For example, the consumption function when ignoring the borrowing constraint
is `cFuncUnc`; normalized market resources that might be achieved next period
are `mNrmNext`; and the marginal marginal value function this period is `vPPfuncNow`.
Additional descriptors beyond abbreviations like this usually use underscores,
e.g. `vFunc_before_PrefShk`.

Variable naming conventions within functions in HARK's numeric tool modules
(like `HARK.interpolation`) is significantly looser than in the model files in
`HARK.ConsumptionSaving`. We want the economic code to be as clear and specific
as reasonably possible: exactly what consumption do you mean here? In contrast,
the numeric methods employed within the numeric tool modules are usually well
known. As long as your code is well commented and variable names are reasonable
in context, it's probably fine.

We strongly encourage you to be liberal with code comments. For each section of
code that does something mathematically or economically describable, put a
comment above it that says what it does, and leave a line break between such
snippets. If there is an "unusual" line of code that deals with an unexpected
issue or complication, leave a short comment in-line.

### Documentation Convention

The HARK team wants the toolKit to be as accessible to users as possible;
our greatest fear (other than spiders, of course) is that a new user will open up our code,
get hopelessly confused trying to read it, and then never look at HARK again.
To prevent this tragic outcome, we have tried hard to provide comprehensive,
accurate documentation and comments within the code describing our methods.
HARK uses the Sphinx utility to generate a website with [online documentation](https://docs.econ-ark.org/)
for all of our tool and model modules, so that users can learn about what's
available in HARK without digging through the source code. Moreover, many of the
example notebooks are automatically rendered to the website. When making contributions
to HARK, the development team asks users to format their inline documentation to
work with Sphinx by following a few simple rules.

- The top of every module should begin with a docstring providing a clear description
  of the contents of the module. The first sentence should concisely summarize the file,
  as it may appear in an index or summary of all modules without the remaining sentences.
  A docstring at the top of a module should be formatted as:

```python
"""
Specifies an economic model and provides methods for solving it.

More specific description of the key features of the model and variations of it in this module.

Maybe some comments about the solution method or limitations of the model.

Your bank account routing number.
"""
```

- The line directly below the declaration of a function, method or class should begin a docstring describing that object.
  As with modules, the first sentence should concisely summarize the function or class,
  as it might be included in an index or summary.
  For functions and methods, the docstring should be formatted as:

```python
def function_name(input1, input2):
    """
    Concise description of the function. More details about what
    the function does, options or modes available, and maybe mathematical
    methods used. Credit to a source if you poached their algorithm.

    Parameters
    ----------

    input1: type
        Description of what input1 represents.
    input2: type
        Description of what input2 represents.

    Returns
    -------
    output_name: type
        Description of the output(s) of the function.  Might have
        multiple entries.  If no output, this is just "None".
    """
```

- Provide ample comments within a function or method so that a relatively
  intelligent reader can follow along with your algorithm. Short comments can
  follow at the end of a line, longer comments (or descriptions of the step
  or task about to be performed) should precede a block of code on the line(s) above it.

Finally, if you write a new model, the HARK team asks that you also provide
a writeup of the model in a Jupyter notebook. The notebook should be put into
`/examples/ModuleName/` and named `AgentTypeSubclassName.ipynb`,
e.g. `/examples/ConsIndShockModel/IndShockConsumerType.ipynb`.
This document does not need to go into great detail about the solution method for
the model or the functions and classes included in the module, but it should:

- Provide a brief textual description of the model, focusing on the feature(s) that
  make it different from other models in HARK.
- Specify the problem mathematically in Bellman form.
- (Optional) Explain how the problem is solved, in broad strokes.
- Provide a mapping from model symbols to names in code.
- (Optional) Describe or summarize the default constructors used.
- Solve one or more examples of the model, maybe just using default parameters, and display policy and/or value functions.

The docstring for an `AgentType` subclass should be substantial and include the following:

- A brief textual description of the model, noting the feature(s) that make it different from other models in HARK.
- The one-period problem stated mathematically in Bellman form.
- (Optional) Describe or summarize the default constructors used.

#### Contributing to Documentation

Econ-ARK's documentation is managed with [Sphinx](https://www.sphinx-doc.org/).
The documentation is written in [reStructuredText](https://www.restructuredtext.org) and
[MyST Markdown](https://myst-parser.readthedocs.io/en/latest/index.html).

Contributing to documentation involves editing the file you want to change,
and creating a pull request in the usual fashion above.
All changes to documentation are automatically rendered and deployed to
`docs.econ-ark.org` by our automated infrastructure.

To test your changes to the documentation locally, you can render as follows:

1. Install the dependencies for the documentation:

   ```bash
    pip install -e .[doc]
    pip install pandoc
   ```

2. Run `sphinx-build`:

  ```bash
      sphinx-build -M html . HARK-docs -T -c docs -W
  ```

3. View the rendered HTML by opening the
   `./HARK-docs/html/index.html` file.

#### Adding examples to the documentation

HARK's example notebooks are stored in the `examples/` directory.
Every pull request to HARK automatically reruns every example notebook
to keep them up to date.

To add a notebook from the examples folder to the documentation, add a link
to the notebook to the `docs/overview/index.rst` file. It should
the format `../../examples/AAA/BBB.ipynb`. Then add a link to the notebook
in the `docs/example_notebooks/Include_list.txt` file. It should have
the format `examples/AAA/BBB.ipynb`.

Sphinx requires its example notebooks to have a title, and headings in order of
decreasing size. Make sure the notebook you added has those qualities before you push.
Otherwise sphinx will fail to build.

:::{warning}
Make sure not to include the HARK-docs folder in your pull request if you're
testing locally. HARK will automatically build this file when the pull request
is made. Including the HARK-docs folder may create unexpected conflicts.

If you would like to build the documentation without warnings being treated as errors use the command:
```bash
       sphinx-build -M html . HARK-docs -T -c docs
```
This lets you see every warning without sphinx quitting after the first issue it finds.
If you're doing this, make sure to delete the HARK-docs folder before running it again.
Otherwise it won't show the warnings again, and you won't be able to check if they've been fixed.
:::

### Testing

`HARK` has a test suite that ensures correct execution on your system.
The test suite has to pass before a pull request can be merged, and tests should
be added to cover any modifications to the code base.

We make use of the [pytest](https://docs.pytest.org/en/latest/) and unittests
testing framework, with all tests located in `/tests/` and its subdirectories,
which match the subdirectory structure of `/HARK/`.

To use `pytest`, ensure that the library is installed in development mode

```bash
$ pip install -e .
```

Now, run all tests using

```bash
$ pytest HARK
```

Or the tests for a specific submodule

```bash
$ pytest HARK/ConsumptionSaving
```

Or tests from a specific file

```bash
$ pytest HARK/ConsumptionSaving/tests/test_ConsAggShockModel.py
```

To view current coverage of HARK's tests, see [here](https://remote-unzip.deno.dev/econ-ark/HARK/main), which is also
linked from the front page of the GitHub repo. The Pytest coverage is automatically
re-run and re-published every Sunday. Some sections of code are hard or
impossible to reach within `pytest` and are thus exempted; this includes platform-specific
code and all `numba` functions that are marked for compilation with `@jit`, as well
as very unusual exceptions that are difficult to build tests for.

### Pre-commit Hooks

`HARK` uses [pre-commit](https://pre-commit.com/) to ensure that all code is well
organized and formatted, as well as to ensure the integrity of example notebooks.
To install `pre-commit`, run

```bash
$ pip install pre-commit
$ pre-commit install
```

Once you do this, it will run the pre-commit hooks on every commit while only
affecting modified files. If you want to run the pre-commit hooks manually on every
file, run

```bash
$ pre-commit run --all-files
```

Because this is an optional feature, it does not come installed with `HARK` by default.
This is to avoid overhead for new users and contributors who might be new to github
and software development in general.

If you are having issues with pre-commit, and just want to commit your changes, you can
use the `--no-verify` flag to bypass the pre-commit hooks.

```bash
$ git commit -m "commit message" --no-verify
```

If you do this, please alert one of the core developers so that we can review your
changes to make sure that there are no issues and that your code is formatted correctly.

The following pre-commit hooks are currently configured:

- [ruff] file formatting
- [pre-commit-hooks] fix end-of-file, trailing whitespace, and requirements.txt

If you are interested in using pre-commit, please see the [pre-commit documentation](https://pre-commit.com/) for more information.

### Pull Request Codes

When you submit a pull request to GitHub, GitHub will ask you for a summary.
If your code is not ready to merge, but you want to get feedback, please consider
using `[WIP] experimental optimization` or similar for the title of your pull
request. That way we will all know that it's not yet ready to merge and that
you may be interested in more fundamental comments about design.

When you think the pull request is ready to merge, change the title (using the
_Edit_ button) to remove the `[WIP]`.

### Bugs

Please [report bugs on GitHub](https://github.com/econ-ark/HARK/issues).

## Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

- (a) The contribution was created in whole or in part by me and I
  have the right to submit it under the open source license
  indicated in the file; or

- (b) The contribution is based upon previous work that, to the best
  of my knowledge, is covered under an appropriate open source
  license and I have the right under that license to submit that
  work with modifications, whether created in whole or in part
  by me, under the same open source license (unless I am
  permitted to submit under a different license), as indicated
  in the file; or

- (c) The contribution was provided directly to me by some other
  person who certified (a), (b) or (c) and I have not modified
  it.

- (d) I understand and agree that this project and the contribution
  are public and that a record of the contribution (including all
  personal information I submit with it, including my sign-off) is
  maintained indefinitely and may be redistributed consistent with
  this project or the open source license(s) involved.
