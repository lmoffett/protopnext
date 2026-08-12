# Getting Started

## Links

- [Source Code](https://github.com/lmoffett/protopnext-prerelease) - `main` branch is our current working branch.

## Contribution Process

During prerelease development, there is a single mainline branch `main`.
All commits to `main` must come through a Pull Request in Github.
Substantial changes need to be tied to Issues in Github.

## Dev Environment

Along with installing the dependencies as described in the [README](./README.md), you will need to install need to setup precommit hooks make contributions, as describe below.

### Setup Git Hooks

Git hooks are configured in `.pre-commit-config.yaml`.
After installing the dependencies, enable the git commit hooks with `pre-commit install`.
**Note:** These hooks will run on every commit, but *only if you do the commit while your project virtual environment is active*.
If the hooks fail to run, it is likely the commit will fail linting in the CI pipeline.

#### Code Formatting and Linting

Currently, the only hook is code formatting with [black](https://pypi.org/project/black/).
Black automatically reformats code to make it [PEP8](https://peps.python.org/pep-0008/) compliant.
Black runs before each commit.
**If black reformats any code, the commit will not occur** and the user will need to readd and commit the reformatted file.

Before committing, you can run `flake8` to check for any existing style issues or `black protopnet` to automatically format the code (these are automatically run by the precommit hook).

## Unit Testing

This project uses [pytest](https://docs.pytest.org/en/7.2.x/) for unit testing.
Tests are python files that end with `_test.py` in the `test` directory.

To run all tests, simply run `pytest`. 
Make sure you have activated your environment.

To run a single test, just add the filename: `pytest test/test_end_to_end.py`.
To run unit tests, run `pytest test`, since unit tests are in the `test` directory.

If you are using `vscode`, you can also configure `pytest` as your `vscode` [test runner](https://code.visualstudio.com/docs/python/testing#_configure-tests).

### Markers

`pytest` [markers](https://docs.pytest.org/en/7.1.x/example/markers.html) can be used to include/exclude test by adding them as annotations to the test definition.
New custom markers can be defined by updating the markers in `pyproject.toml` under `tool.pytest.ini_options` > `markers`.
For instance, to run tests that require CUDA, use `pytest -m cuda`.

## Continuous Integration

TODO