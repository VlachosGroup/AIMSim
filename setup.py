import AIMSim
import pathlib
from setuptools import setup, find_packages

cwd = pathlib.Path(__file__).parent

README = (cwd / "README.md").read_text()

desc = "Python command line and GUI tool to analyze molecular similarity."

vers = AIMSim.__version__

setup(
    name="AIMSim",
    version=vers,
    description=desc,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/VlachosGroup/AIMSim",
    author="Himaghna Bhattacharjee, Jackson Burns",
    license="MIT",
    classifiers=["Programming Language :: Python :: 3"],
    install_requires=[
        "scipy==1.5.4",
        "matplotlib==3.3.4",
        "seaborn==0.11.1",
        "tabulate==0.8.9",
        "numpy==1.21.0",
        "multiprocess==0.70.12.2",
        "scikit_learn_extra==0.2.0",
        "pandas==1.1.5",
        "mordred==1.2.0",
        "PyYAML==5.4.1",
        "scikit_learn==0.24.2",
        "networkx==2.1",
        "rdkit-pypi",
        "psutil",
        "padelpy",
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "AIMSim=AIMSim.__main__:start_AIMSim",
        ]
    },
)
