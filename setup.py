import os.path
import codecs
import pathlib
from setuptools import setup, find_packages

cwd = pathlib.Path(__file__).parent

README = (cwd / "README.md").read_text()

desc = "Python command line and GUI tool to analyze molecular similarity."


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="aimsim",
    python_requires=">=3.6",
    version=get_version("aimsim/__init__.py"),
    description=desc,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/VlachosGroup/AIMSim",
    author="Himaghna Bhattacharjee, Jackson Burns",
    license="MIT",
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=read("requirements.txt").split("\n"),
    extras_require={
        "mordred": [
            "mordred==1.2.0",
            "networkx==2.*",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "aimsim=aimsim.__main__:start_AIMSim",
        ]
    },
)
