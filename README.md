# **SigmaEpsilon.Solid.Material** - Solid Mechanics in Python

![ ](logo.png)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sigma-epsilon/sigmaepsilon.solid.material/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/sigma-epsilon/sigmaepsilon.solid.material/tree/main)
[![codecov](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.solid.material/graph/badge.svg?token=7JKJ3HHSX3)](https://codecov.io/gh/sigma-epsilon/sigmaepsilon.solid.material)
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilonsolidmaterial/badge/?version=latest)](https://sigmaepsilonsolidmaterial.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/sigmaepsilon.solid.material.svg)](https://pypi.org/project/sigmaepsilon.solid.material)
[![Python 3.7‒3.10](https://img.shields.io/badge/python-3.7%E2%80%923.10-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Requirements Status](https://dependency-dash.repo-helper.uk/github/sigma-epsilon/sigmaepsilon.solid.material/badge.svg)](https://dependency-dash.repo-helper.uk/github/sigma-epsilon/sigmaepsilon.solid.material)

> **Note**
> Here and there, implementation of the performance critical parts of the library rely on the JIT-compilation capabilities of Numba. This means that the library performs well even for large scale problems, on the expense of a longer first call.

## Highlights

* Classes to handle linear elastic materials of all kinds.

## Documentation

The [documentation](https://sigmaepsilonsolidmaterial.readthedocs.io/en/latest/) is built with [Sphinx](https://www.sphinx-doc.org/en/master/) using the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html) and hosted on [ReadTheDocs](https://readthedocs.org/). Check it out for the user guide, an ever growing set of examples, and API Reference.

## Installation

sigmaepsilon.solid.material can be installed from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install sigmaepsilon.solid.material
```

or chechkout with the following command using GitHub CLI

```console
gh repo clone sigma-epsilon/sigmaepsilon.solid.material
```

and install from source by typing

```console
>>> pip install .
```

If you want to run the tests, you can install the package along with the necessary optional dependencies like this

```console
>>> pip install ".[test]"
```

### Development mode

If you are a developer and want to install the library in development mode, the suggested way is by using this command:

```console
>>> pip install "-e .[test, dev]"
```

### Checking your installation

You should be able to import sigmaepsilon.mesh from the Python prompt:

```console
$ python
Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import sigmaepsilon.solid.material
>>> sigmaepsilon.solid.material.__version__
'2.2.0'
```

## Testing and coverage

The following command runs all tests and creates a html report in a folder named `htmlcov` (the settings are governed by the `.coveragerc` file):

```console
python -m pytest --cov-report html --cov-config=.coveragerc --cov sigmaepsilon.mesh
```

Open `htmlcov/index.html` to see the results.

## Changes and versioning

See the [changelog](CHANGELOG.md), for the most notable changes between releases.

The project adheres to [semantic versioning](https://semver.org/).

## How to contribute?

Contributions are currently expected in any the following ways:

* finding bugs
  If you run into trouble when using the library and you think it is a bug, feel free to raise an issue.
* feedback
  All kinds of ideas are welcome. For instance if you feel like something is still shady (after reading the user guide), we want to know. Be gentle though, the development of the library is financially not supported yet.
* feature requests
  Tell us what you think is missing (with realistic expectations).
* examples
  If you've done something with the library and you think that it would make for a good example, get in touch with the developers and we will happily inlude it in the documention.
* sharing is caring
  If you like the library, share it with your friends or colleagues so they can like it too.

In all cases, read the [contributing guidelines](CONTRIBUTING.md) before you do anything.

## Acknowledgements

Although `sigmaepsilon.mesh` works without `VTK` or `PyVista` being installed, it is highly influenced by these libraries and works best with them around. Also shout-out for the developers of `NumPy`, `Scipy`, `Numba`, `Awkward`, `meshio` and all the third-party libraries involved in the project. Whithout these libraries the concept of writing performant, yet elegant Python code would be much more difficult.

**A lot of the packages mentioned on this document here and the introduction have a citable research paper. If you use them in your work through sigmaepsilon.mesh, take a moment to check out their documentations and cite their papers.**

Also, funding of these libraries is partly based on the size of the community they are able to support. If what you are doing strongly relies on these libraries, don't forget to press the :star: button to show your support.

## License

This package is licensed under the [MIT license](LICENSE.txt).
