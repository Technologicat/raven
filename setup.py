#!/usr/bin/env python

import ast
import os

from setuptools import setup, find_packages  # type: ignore[import]


def read(*relpath, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *relpath),
              encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()

def readlines(*relpath, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *relpath),
              encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.readlines()

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
init_py_path = os.path.join("raven", "__init__.py")
version = None
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith("__version__"):
                module = ast.parse(line, filename=init_py_path)
                expr = module.body[0]
                assert isinstance(expr, ast.Assign)
                v = expr.value
                if type(v) is ast.Constant:
                    # mypy understands `isinstance(..., ...)` but not `type(...) is ...`,
                    # and we want to match on the exact type, not any subclass that might be
                    # added in some future Python version.
                    assert isinstance(v, ast.Constant)
                    version = v.value
                break
except FileNotFoundError:
    pass
if not version:
    raise RuntimeError(f"Version information not found in {init_py_path}")

# TODO: As of 0.1.0, just making this installable for now; fix package metadata later.
setup(
    name="raven",
    version=version,
    packages=["raven", "raven.fonts", "raven.vendor", "raven.vendor.DearPyGui_Markdown", "raven.vendor.file_dialog", "raven.vendor.file_dialog.images"],
    package_data={"": ["*.ttf", "*.png"]},
    provides=["raven"],
    keywords=["visualization", "semantic-analysis", "application", "AI", "NLP", "GUI"],
    install_requires=readlines("requirements.txt"),
    python_requires=">=3.10",
    author="Juha Jeronen and JAMK University of Applied Sciences",
    author_email="juha.jeronen@jamk.fi",
    url="https://github.com/Technologicat/raven",
    description="Semantic visualizer and analyzer",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="BSD",
    platforms=["Linux"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities"
    ],
    entry_points={"console_scripts": ["raven-visualizer=raven.app:main"]},
    zip_safe=True
)
