# Time Series Analysis

A framework for time-series analysis and features extraction / visualization.

## Installation

```
pip install --user -e .[dev]
```

## Contribute

To contribute to this repo, please follow this steps:
1. create or assigne yourself an open issue
2. go to the develop branch and pull it : `git checkout develop` and `git pull`
3. create a new branch with the issue id: `git checkout -b #123-this-feature`
4. do as much as commit as needed, then push your changes.
5. [optional] write a test in `tests` dir
6. once your implementation is finished:
    - format the code: `invoke format` or `black tsanalysis`
    - check your code format: `invoke lint` or `pylint tsanalysis`
    - test it: `invoke test` or `pytest tests`
7. [optional] fix/commit/push the changes
8. **document** your code ! (cf *Documentation* below)
9. ask for a pull request

## Documentation

### Browse documentation

To build and browse the documentation, go to the root folder and execute:

```
invoke doc
```

### Contribute

The code's documentation is written following [numpy doc standards](https://numpydoc.readthedocs.io/en/latest/format.html). Please find a class example [here](https://numpydoc.readthedocs.io/en/latest/example.html#example).
