<h1 align="center">AIMSim README</h1> 
<h3 align="center">Visualizing Diversity in your Molecular Dataset</h3>

![AIMSim Logo](interfaces/UI/AIMSim-logo.png)
<p align="center">
  <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/VlachosGroup/AIMSim?style=social">
  <img alt="Total Downloads" src="https://static.pepy.tech/personalized-badge/aimsim?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads">
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/aimsim">
  <img alt="commits since" src="https://img.shields.io/github/commits-since/VlachosGroup/AIMSim/latest.svg">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/aimsim">
  <img alt="PyPI - License" src="https://img.shields.io/github/license/VlachosGroup/AIMSim">
</p>

## Documentation and Tutorial
[View our Online Documentation](https://vlachosgroup.github.io/AIMSim/) or try the [_AIMSim_ comprehensive tutorial in your browser](https://mybinder.org/v2/gh/VlachosGroup/AIMSim/master?labpath=AIMSim-demo.ipynb).

## Purpose

__Why Do We Need To Visualize Molecular Similarity / Diversity?__

There are several contexts where it is helpful to visualize the diversity of a molecular dataset:

_Exploratory Experimental Synthesis_

For a chemist, synthesizing new molecules with targeted properties is often a laborious and time consuming task.
In such a case, it becomes useful to check the similarity of a newly proposed (un-synthesized) molecule to the ones already synthesized.
If the proposed molecule is too similar to the existing repertoire of molecules, it will probably not yield not enough new information /
property and thus need not be synthesized. Thus, a chemist can avoid spending
time and effort synthesizing molecules not useful for the project.

_Lead Optimization and Virtual Screening_

This application is the converse of exploratory synthesis where the interest is to find molecules in a database which are structurally similar to an "active" molecule. In this context, "active" might refer to pharmocological activity (drug discover campaigns) or desirable chemical properties (for example, to discover alternative chemicals and solvents for an application). In such a case, AIMSim helps to run virtual screenings over a molecular database and visualize the results.

_Machine Learning Molecular Properties_

In the context of machine learning, visualizing the diversity of the training set gives a good idea about its information quality.
A more diverse training data-set yields a more robust model, which generalizes well to unseen data. Additionally, such a visualization can 
identify "clusters of similarity" indicating the need for separately trained models for each cluster.

_Substrate Scope Robustness Verification_

When proposing a novel reaction it is essential for the practicing chemist to evaluate the transformation's tolerance of diverse functional groups and substrates (Glorius, 2013). Using `AIMSim`, one can evaluate the structural and chemical similarity across an entire susbtrate scope to ensure that it avoids redundant species. Below is an example similarity heatmap generated to visualize the diversity of a three-component sulfonamide coupling reaction with a substantial number of substrates (Chen, 2018).
![Image of sulfonamide substrate scope](tests/sulfonamide-substrate-scope.png)

Many of the substrates appear similar to one another and thereby redundant, but in reality the core sulfone moiety and the use of the same coupling partner when evaluating functional group tolerance accounts for this apparent shortcoming. Also of note is the region of high similarity along the diagonal where the substrates often differ by a single halide heteratom or substitution pattern.

## Installing AIMSim
`AIMSim` can be installed with a single command:
`pip install aimsim`
This command also installs the required dependencies. It is recommended to install `AIMSim` in a separate virtual environment.

_Optional:_ AIMSim can also use descriptors provided in the `mordred` package, but because it is an older package it is only compatible with Python 3.8. To install with `mordred` support, use `pip install aimsim[mordred]`

## Running AIMSim
`AIMSim` is compatible with Python 3.7 to 3.9.
Start `AIMSim` with a graphical user interface:

`aimsim`

Start `AIMSim` with a prepared configuration YAML file (`config.yaml`):

`aimsim config.yaml`

### Currently Implemented Fingerprints

1. Morgan Fingerprint (Equivalent to the ECFP fingerprints)
2. RDKit Topological Fingerprint
3. RDKit Daylight Fingerprint
4. All fingerprints available from the [ccbmlib](https://github.com/vogt-m/ccbmlib) package (_specify 'ccbmlib:descriptorname' for command line input_).
5. All descriptors available through the [Mordred](https://github.com/mordred-descriptor/mordred) library (_specify 'mordred:desciptorname' for command line input._). To enable this option, you must install with `pip install aimsim['mordred']`.
6. All descriptors and fingerprints available from [PaDELPy](https://github.com/ecrl/padelpy), an interface to PaDEL-Descriptor. (_specify 'padelpy:desciptorname' for command line input._).

### Currently Implemented Similarity Scores

44 commonly used similarity scores are implemented in AIMSim.
Additional L0, L1 and L2 norm based similarities are also implemented. [View our Online Documentation](https://vlachosgroup.github.io/AIMSim/implemented_metrics.html) for a complete list of implemented similarity scores.


### Currently Implemented Functionalities

1. Measure Search: Automate the search of fingerprint and similarity metric (called a "measure") using the following algorithm:
  Step 1: Select an arbitrary featurization scheme.
  Step 2: Featurize the molecule set using the selected scheme.
  Step 3: Choose an arbitrary similarity measure.
  Step 4: Select each molecule’s nearest and furthest neighbors in the set using the similarity measure.
  Step 5: Measure the correlation between a molecule’s QoI and its nearest neighbor’s QoI.
  Step 6: Measure the correlation between a molecule’s QoI and its further neighbor’s QoI.
  Step 7: Define a score which maximizes the value in Step 5 and minimizes the value in Step 6.
  Step 8: Iterate Steps 1 – 7 to select the featurization scheme and similarity measure to maximize the result of Step 7. 
2. See Property Variation with Similarity: Visualize the correlation in the QoI between nearest neighbor molecules (most similar pairs in the molecule set) and between the furthest neighbor molecules (most dissimilar pairs in the molecule set). This is used to verify that the chosen measure is appropriate for the task.

3. Visualize Dataset: Visualize the diversity of the molecule set in the form of a pairwise similarity density and a similarity heatmap of the molecule set. Embed the molecule set in 2D space using using principal component analysis (PCA)[3], multi-dimensional scaling[4], t-SNE[5], Spectral Embedding[6], or Isomap[7].

4. Compare Target Molecule to Molecule Set: Run a similarity search of a molecule against a database of molecules (molecule set). This task can be used to identify the most similar (useful in virtual screening operations) or most dissimilar (useful in application that require high diversity such as training set design for machine learning models) molecules.

5. Cluster Data: Cluster the molecule set. The following algorithms are implemented: 

For arbitrary molecular features or similarity metrics with defined Euclidean distances: K-Medoids[3] and Ward[8] (hierarchical clustering).

For binary fingerprints: Complete, single and average linkage hierarchical clustering[8].

The clustered data is plotted in two dimensions using principal component analysis (PCA)[3], multi-dimensional scaling[4], or TSNE[5].

6. Outlier Detection: Using an isolation forest, check for which molecules are potentially novel or are outliers according to the selected descriptor. Output can be directly to the command line by specifiying `output` to be `terminal` or to a text file by instead providing a filename.

## Contributors

Developer: Himaghna Bhattacharjee, Vlachos Research Lab. ([LinkedIn](www.linkedin.com/in/himaghna-bhattacharjee))

Developer: Jackson Burns, Don Watson Lab. ([Personal Site](https://www.jacksonwarnerburns.com/))

## Developer Notes
Issues and Pull Requests are welcomed! To propose an addition to `AIMSim` open an issue and the developers will tag it as an _enhancement_ and start discussion.

`AIMSim` includes an automated testing apparatus operated by Python's _unittest_ built-in package. To execute tests related to the core functionality of `AIMSim`, run this command:

`python -m unittest discover`

Full multiprocessing speedup and efficiency tests take more than 10 hours to run due to the number of replicates required. To run these tests, create a file called `.speedup-test` in the `AIMSim` directory and execute the above command as shown.

To manually build the docs, execute the following with `sphinx` and `m2r` installed and from the `/docs` directory:

`m2r ../README.md | mv ../README.rst . | sphinx-apidoc -f -o . .. | make html | cp _build/html/* .`

Documentation should manually build on push to master branch via an automated GitHub action.

For packaging on PyPI:

`python -m build; twine upload dist/*`

Be sure to bump the version in `__init__.py`.

## Citation
If you use this code for scientific publications, please cite the following paper.

Bhattacharjee, H.҂; Burns, J.҂; Vlachos, D.G. (2021): AIMSim: An Accessible Cheminformatics Platform for Similarity Operations on Chemicals Datasets. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv-2022-nw6f5

## License
This code is made available under the terms of the _MIT Open License_:

Copyright (c) 2020 Himaghna Bhattacharjee & Jackson Burns

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Works Cited
[1] Collins, K. and Glorius, F., A robustness screen for the rapid assessment of chemical reactions. Nature Chem 5, 597–601 (2013). https://doi.org/10.1038/nchem.1669

[2] Chen, Y., Murray, P.R.D., Davies, A.T., and Willis M.C., J. Am. Chem. Soc. 140 (28), 8781-8787 (2018). https://doi.org/10.1021/jacs.8b04532

[3] Hastie, T., Tibshirani R. and Friedman J., The Elements of statistical Learning: Data Mining, Inference, and Prediction, 2nd Ed., Springer Series in Statistics (2009).

[4] Borg, I. and Groenen, P.J.F., Modern Multidimensional Scaling: Theory and Applications, Springer Series in Statistics (2005).

[5] van der Maaten, L.J.P. and Hinton, G.E., Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9:2579-2605 (2008).

[6] Ng, A.Y., Jordan, M.I. and Weiss, Y., On Spectral Clustering: Analysis and an algorithm. ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, MIT Press (2001).

[7] Tenenbaum, J.B., De Silva, V. and Langford, J.C, A global geometric framework for nonlinear dimensionality reduction, Science 290 (5500), 2319-23 (2000). https://doi.org/10.1126/science.290.5500.2319.

[8] Murtagh, F. and Contreras, P., Algorithms for hierarchical clustering: an overview. WIREs Data Mining Knowl Discov (2011). https://doi.org/10.1002/widm.53



