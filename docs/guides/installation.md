# HARK installation guide

HARK is an open source project written in Python. It's supported for Python versions 3.10 and up.

## Instructions for a new user

In order to use HARK, you firstly need to download Python and add Python to the PATH. If you are not confident about the installation process you can use [this step-by-step guide](https://realpython.com/installing-python/). You can also install the Anaconda Python distribution by following [this](https://docs.anaconda.com/free/anaconda/install/) guide. Anaconda comes bundled with most commonly used mathematical and scientific Python packages. We recommend using Conda to all HARK users.

:::{note}

You can have the default Python distribution from python.org and from Anaconda, as they do not interfere. However, be careful with setting the PATH. To avoid problems you can set the environment variables path to the default distribution and access Anaconda distribution via Anaconda prompt.
:::

Next, you'll want a text editor for Python. We recommend using [VSCode](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/). If you're using Anaconda, we'd also recommend using Spyder which comes bundled.

``````{tab-set}
`````{tab-item} VScode
To install VScode visit:
[https://code.visualstudio.com/docs](https://code.visualstudio.com/docs)
`````
`````{tab-item} Pycharm
To install Pycharm visit:
[https://www.jetbrains.com/help/pycharm/installation-guide.html](https://www.jetbrains.com/help/pycharm/installation-guide.html)
`````
`````{tab-item} Anaconda-Spyder

To install Anaconda, follow the guide [here](https://docs.anaconda.com/free/anaconda/install/).

They may ask you to give them your email to install Anaconda. If they do you can click the skip registration button and it will take you directly to the installation window.

Once Anaconda is installed visit the [Sypder installation guide](https://docs.spyder-ide.org/current/installation.html#conda-based-distributions) to set up Spyder.

Sypder can be opened through the Anaconda navigator, or by typing `spyder` into the command line.

`````
`````{tab-item} Other Options

If you're looking for more options or these recommendations aren't working for you, don't worry, there are plenty of others. You can start with [This](https://wiki.python.org/moin/PythonEditors) list or just type 'Python IDE' into your prefered search engine.

`````
``````

After installing Python and the text editor, you can install the HARK package. The simplest way is to use [pip](https://pip.pypa.io/en/stable/installing/) by running the command `pip install econ-ark`.

If you prefer to isolate your installation of `econ-ark` from the installations of any other python tools and packages, use a virtual environment such as [virtualenv](https://virtualenv.pypa.io/en/latest/).

``````{tab-set}
`````{tab-item} Using virtualenv

To install `virtualenv`, then to create an environment named `econ-ark`, and finally to activate that environment, at a command line type:

```
cd [directory where you want to store the econ-ark virtual environment]
pip install virtualenv
virtualenv econ-ark
```

Then for Windows, type:

```
.\econ-ark\Scripts\activate.bat

```

While for Mac or Linux:

```
source econ-ark/bin/activate
```

In the both cases, you may see (econ-ark) in your command prompt.

If you want to leave your environment, type:
```
deactivate
```
:::{note}

If you install econ-ark into a virtual environment, your HARK scripts will not compile unless the virtual environment is activated.

If you're using an IDE, you also need to configure the environment.

check out the virtual env [documentation](https://virtualenv.pypa.io/en/latest/user_guide.html) for more information about virtual environments.
:::
`````
`````{tab-item} Using Conda
Anaconda includes its own virtual environment system called `conda` which stores environments in a preset location (so you don't have to choose). So in order to create your econ-ark virtual environment type:

```
conda create -n econ-ark anaconda
```
Then you can activate it by typing:
```
conda activate econ-ark
```
If you want to leave your environment, type:
```
deactivate
```
:::{note}

If you install econ-ark into a virtual environment, your HARK scripts will not compile unless the virtual environment is activated.

If you're using an IDE, you also need to configure the environment.

check out the conda [documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information about virtual environments.
:::
`````
``````

Next, install `econ-ark` into your new virtual environment via pip:

```
pip install econ-ark
```

---
## Instructions for an advanced user/developer

If you want to make changes or contributions (yay!) to HARK, you'll need to have access to the source files. Installing HARK via pip (either at the command line, or inside Spyder) makes it hard to access those files (and it's a bad idea to mess with the original code anyway because you'll likely forget what changes you made). If you are adept at GitHub, you can [fork](https://help.github.com/en/articles/fork-a-repo) the repo. If you are less experienced, you should download a personal copy of HARK again using `git clone` (see part for a new user) or the [GitHub Desktop app](https://desktop.github.com/).

1.  Create and activate a virtual environment.

2.  Navigate to wherever you want to put the repository and type `git clone git@github.com:econ-ark/HARK.git` ([more details here](https://git-scm.com/doc)). If you get a permission denied error, you may need to setup SSH for GitHub, or you can clone using HTTPS: `git clone https://github.com/econ-ark/HARK.git`. If you're using the desktop app, go to file->Clone Repository->url, and use `https://github.com/econ-ark/HARK.git` as the Repository url.

3.  Make sure to navigate to the HARK directory, and install HARK's requirements into the virtual environment with `pip install -r ./requirements/base.txt`. The requirements folder also contains `dev.txt` and `doc.txt` which install packages relevant to develoment and the documentation respectively.

4.  Install econ-ark locally: navigate to your local repository and then type:

```
pip install -e .
```
5.  To check that everything has been set up correctly, run HARK's tests with `python -m unittest`.

---

:::{note}
To check the package performance with your local changes, use the same command from the command line (after navigating to your HARK directory):

```
pip install -e .
```

If for some reason you want to switch back to the PyPI version:

```
pip uninstall econ-ark
pip install econ-ark (options)
```
:::

## Content Aside from the Toolkit

If you don't already have one, you should designate a place on your computer where you will be putting all the Econ-ARK content.
For example, it is common to just create a folder `GitHub` in your home user directory. Inside `GitHub` you should create folders
corresponding to the GitHub ID's of users whose work you want to obtain; for example, `GitHub/econ-ark`. Inside the `econ-ark` directory you can obtain a number of different resources.

### Examples, Demonstrations, And Illustrations

To copy HARK's example and documentation notebooks into a local working directory, run the following two lines in a Python environment:

```python
from HARK import install_examples
install_examples()
```

You will be prompted to choose a local directory into which an examples subdirectory will be created. This will contain many Jupyter notebooks with guides, examples, and documentation for HARK.

To look at demonstrations, check repository: \* [DemARK](https://github.com/econ-ark/DemARK): Demonstrations of the use of HARK

You will want to obtain your own local copy of these repos using:

```
git clone https://github.com/econ-ark/DemARK.git
```

Once you have downloaded them, you will find that the repo contains a `notebooks` directory that contains a number of [jupyter notebooks](https://jupyter.org/). If you have the jupyter notebook tool installed (it is installed as part of Anaconda), you should be able to launch the jupyter notebook app from the command line with the command: `jupyter notebook`.

### [REMARK](https://github.com/econ-ark/REMARK/#readme): Replications and Examples Made Using the ARK

From inside the `GitHub/econ-ark` directory, you will want to obtain your own local copy of the REMARK repo:

```
git clone https://github.com/econ-ark/REMARK.git
```

Once the download finishes (it should have created `GitHub/econ-ark/REMARK`, change into that directory.
Its root level is mostly descriptive; the main content is in the `REMARKs` subdirectory, so `cd REMARKs` to
have a look at what is there. Each REMARK is contained in a directory with the handle of the REMARK;
for example, `BufferStockTheory` is the handle for the REMARK on 'The Theoretical Foundations of Buffer Stock Saving'.

At the top level of the directory for each REMARK we have some meta-information (title, authors, etc) and an eponymous Jupyter notebook, e.g. `BufferStockTheory.ipynb.` In most cases there will also be an subdirectory, e.g. `BufferStockTheory` which is a placeholder for the substantive content of the project (like, the original paper).

- Until a REMARK is finalized and frozen, the original content is often kept somewhere else, e.g.\ in an author's GitHub repo. In this case, the REMARK repo uses a `submodule` version, which is sort of like a link to the original material. To save space, a regular 'clone' of the REMARK repo does not incorporate all the submodules; therefore you may find those folders empty when you first use them. In order to obtain the **real** content, in the root directory of the repo in question (e.g., in REMARKs/BufferStockTheory), you need execute the command `git submodule update --recursive --remote`.

Once you have downloaded them, you will find that the repo contains a `notebooks` directory that contains a number of [jupyter notebooks](https://jupyter.org/). If you have the jupyter notebook tool installed (it is installed as part of Anaconda), you should be able to launch the jupyter notebook app from the command line with the command:

```
jupyter notebook
```

## Example: Installing HARK and Running a Simple Model

To help new users get started with HARK, let's walk through an example of installing HARK and running a simple model.

### Step 1: Install Python and a Text Editor

First, make sure you have Python installed on your computer. You can download Python from the official [Python website](https://www.python.org/downloads/). Follow the installation instructions for your operating system.

Next, install a text editor for writing and running Python code. We recommend using [VSCode](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/). If you're using Anaconda, you can also use Spyder, which comes bundled with Anaconda.

### Step 2: Create a Virtual Environment

To keep your HARK installation isolated from other Python packages, create a virtual environment. You can use either `virtualenv` or `conda` for this purpose.

#### Using virtualenv

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to store the virtual environment.
3. Run the following commands:

```
pip install virtualenv
virtualenv econ-ark
```

4. Activate the virtual environment:

- For Windows:

```
.\econ-ark\Scripts\activate.bat
```

- For Mac or Linux:

```
source econ-ark/bin/activate
```

#### Using Conda

1. Open a terminal or command prompt.
2. Run the following commands:

```
conda create -n econ-ark anaconda
conda activate econ-ark
```

### Step 3: Install HARK

With the virtual environment activated, install HARK using `pip`:

```
pip install econ-ark
```

### Step 4: Run a Simple Model

Now that HARK is installed, let's run a simple model. Create a new Python file (e.g., `simple_model.py`) and add the following code:

```python
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType

# Define the parameters for the model
params = {
    "CRRA": 2.5,  # Relative risk aversion
    "DiscFac": 0.96,  # Discount factor
    "Rfree": 1.03,  # Risk-free interest factor
    "LivPrb": [0.98],  # Survival probability
    "PermGroFac": [1.01],  # Income growth factor
    "T_cycle": 1,
    "cycles": 0,
    "AgentCount": 10000,
}

# Create an instance of the model
model = PerfForesightConsumerType(**params)

# Solve the model
model.solve()

# Print the consumption function
print(model.solution[0].cFunc)
```

Save the file and run it from the terminal or command prompt:

```
python simple_model.py
```

You should see the consumption function printed in the output.

Congratulations! You've successfully installed HARK and run a simple model. For more examples and detailed explanations, refer to the [HARK documentation](https://docs.econ-ark.org/).

## Additional Examples and Tutorials

To help new users get started with the repository more easily, we have added more detailed explanations and examples in the following sections:

- [Overview and Examples](https://docs.econ-ark.org/docs/overview/index.html): This section provides an introduction to HARK and includes various examples to help users understand how to use the toolkit.
- [Guides](https://docs.econ-ark.org/docs/guides/index.html): This section includes guides on installation, quick start, and contributing to HARK.
- [Reference](https://docs.econ-ark.org/docs/reference/index.html): This section provides detailed explanations and examples of the various tools and models available in the repository.

For more information and resources, please visit the [Econ-ARK documentation](https://docs.econ-ark.org/).
