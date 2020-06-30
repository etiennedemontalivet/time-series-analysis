# AI Python Analysis

## Installation

### Create virtual env

We recommend to create virtual env to use the framework. Please visit [this page](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to do so.

### Installing the package:

Once you are inside your environment, you can install the command using pip:

```
pip install -e .[dev]
```


## Ensuring code quality

In order to ensure the best code quality, two tools are available:

#### Use `black` to format code

You can use `black` to format the code. Simply run:

```
black framework
```

####  Using `flake8` to lint code

You can use `flake8` to lint the code. Simply run:

```
flake8
```

## Documentation

This is **Work In Progress**


A documentation can be generated automatically from docstrings found in the code.
You can try it:

Linux:
```
cd docs
make html
```

This will build an HTML documentation inside `docs\build` directory.
You can visualize it on `http://localhost:8000` running the following command:

```
cd docs\build
python -m http.server 8000
```
