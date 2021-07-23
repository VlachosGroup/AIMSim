import pathlib
from setuptools import setup, find_packages

# The directory containing this file
cwd = pathlib.Path(__file__).parent

# The text of the README file
README = (cwd / "README.md").read_text()

try:
    from rdkit import Chem
except ImportError:
    # raise BaseException(
    #     "RDKit installation not found! Run `conda install -c rdkit rdkit`.")
    pass

# This call to setup() does all the work
setup(
    name="molSim",
    version="0.0.1",
    description="Python command line and GUI tool to analyze molecular similarity.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/himaghna/molSim",
    author="Himaghna Bhattacharjee, Jackson Burns",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    install_requires=[
        "python==3.7.10",
        "scipy==1.5.4",
        "matplotlib==3.3.4",
        "seaborn==0.11.1",
        "tabulate==0.8.9",
        "numpy==1.19.5",
        "multiprocess==0.70.12.2",
        "scikit_learn_extra==0.2.0",
        "pandas==1.1.5",
        "mordred==1.2.0",
        "PyYAML==5.4.1",
        "scikit_learn==0.24.2",
        "networkx==2.1",
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'molSim=molSim.__main__:start_molSim',
        ]},
)
