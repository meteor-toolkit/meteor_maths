import io
import os
import re

from setuptools import find_packages
from setuptools import setup

exec(open("matheo/_version.py").read())


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    version=__version__,
    name="matheo",
    url="https://github.com/meteor-toolkit/matheo",
    license="None",
    author="Sam Hunt, Pieter De Vis",
    author_email="sam.hunt@npl.co.uk",
    description="Matheo is a python package with mathematical algorithms for use in Earth observation data and tools",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.10",
    install_requires=["pyspectral", "punpy>=1.0.7", "pyproj"],
    extras_require={
        "dev": [
            "pre-commit",
            "tox",
            "sphinx",
            "sphinx_design",
            "sphinx_book_theme",
            "ipython",
            "sphinx_autosummary_accessors",
        ],
    },
)
