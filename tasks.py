"""
#TODO: This file assumes that the developer uses a Linux environment and Pip. It should be updated to work with Conda and Windows.
This is an equivalent of Makefile that works both on Linux and Windows.
Check invoke documentation on http://www.pyinvoke.org.

It can be used to:

- build package
- build documentation
- serve documentation
"""
import os
from invoke import task


@task
def clean(c, docs=True, extra=''):
    patterns = ['build', 'dist']
    if docs:
        patterns.append('docs/build')
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        print(pattern)
        c.run(f"rm -Rf {pattern}")


@task
def build(c, docs=False):
    c.run("python setup.py sdist bdist_wheel")
    if docs:
        c.run("cd docs && make html")


@task
def servedocs(c, port=8000):
    c.run(
        f"cd docs && make html && cd build/html && python -m http.server {port}", pty=True)


@task
def format(c):
    c.run("black tsanalysis")

@task
def test(c):
    c.run("python -m pytest")


@task
def lint(c):
    c.run("pylint tsanalysis")


@task
def install(c):
    c.run("pip install --user -e .[dev]")
