.. role:: raw-html-m2r(raw)
   :format: html


:raw-html-m2r:`<h1 align="center">AIMSim README</h1>` 


.. raw:: html

   <h3 align="center">Visualizing Diversity in your Molecular Dataset</h3>



.. image:: interfaces/UI/AIMSim-logo.png
   :target: interfaces/UI/AIMSim-logo.png
   :alt: AIMSim Logo



.. raw:: html

   <p align="center">
     <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/VlachosGroup/AIMSim?style=social">
     <img alt="commits since" src="https://img.shields.io/github/commits-since/VlachosGroup/AIMSim/latest.svg">
     <img alt="PyPI" src="https://img.shields.io/pypi/v/aimsim">
     <img alt="PyPI - License" src="https://img.shields.io/github/license/VlachosGroup/AIMSim">
     <img alt="Test Status" src="https://github.com/VlachosGroup/AIMSim/actions/workflows/ci.yml/badge.svg?event=schedule">
   </p>


Repository Status: 
.. image:: https://www.repostatus.org/badges/latest/inactive.svg
   :target: https://www.repostatus.org/#inactive
   :alt: Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.


AIMSim has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.

Downloads Stats:


* `aimsim`: [![Downloads](https://static.pepy.tech/badge/aimsim)](https://static.pepy.tech/personalized-badge/aimsim?period=total&units=none&left_color=grey&right_color=blue&left_text=Lifetime%20Downloads)
* `aimsim_core`: [![Downloads](https://static.pepy.tech/badge/aimsim_core)](https://pepy.tech/project/aimsim_core?period=total&units=none&left_color=grey&right_color=blue&left_text=Lifetime%20Downloads)

Documentation and Tutorial
--------------------------

`View our Online Documentation <https://vlachosgroup.github.io/AIMSim/>`_ or try the *AIMSim* comprehensive tutorial in your browser:
:raw-html-m2r:`<a target="_blank" href="https://colab.research.google.com/github/VlachosGroup/AIMSim/blob/master/AIMSim-demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>`

Purpose
-------

**Why Do We Need To Visualize Molecular Similarity / Diversity?**

There are several contexts where it is helpful to visualize the diversity of a molecular dataset:

*Exploratory Experimental Synthesis*

For a chemist, synthesizing new molecules with targeted properties is often a laborious and time consuming task.
In such a case, it becomes useful to check the similarity of a newly proposed (un-synthesized) molecule to the ones already synthesized.
If the proposed molecule is too similar to the existing repertoire of molecules, it will probably not yield not enough new information /
property and thus need not be synthesized. Thus, a chemist can avoid spending
time and effort synthesizing molecules not useful for the project.

*Lead Optimization and Virtual Screening*

This application is the converse of exploratory synthesis where the interest is to find molecules in a database which are structurally similar to an "active" molecule. In this context, "active" might refer to pharmocological activity (drug discover campaigns) or desirable chemical properties (for example, to discover alternative chemicals and solvents for an application). In such a case, AIMSim helps to run virtual screenings over a molecular database and visualize the results.

*Machine Learning Molecular Properties*

In the context of machine learning, visualizing the diversity of the training set gives a good idea about its information quality.
A more diverse training data-set yields a more robust model, which generalizes well to unseen data. Additionally, such a visualization can 
identify "clusters of similarity" indicating the need for separately trained models for each cluster.

*Substrate Scope Robustness Verification*

When proposing a novel reaction it is essential for the practicing chemist to evaluate the transformation's tolerance of diverse functional groups and substrates (Glorius, 2013). Using ``AIMSim``\ , one can evaluate the structural and chemical similarity across an entire susbtrate scope to ensure that it avoids redundant species. Below is an example similarity heatmap generated to visualize the diversity of a three-component sulfonamide coupling reaction with a substantial number of substrates (Chen, 2018).

.. image:: tests/sulfonamide-substrate-scope.png
   :target: tests/sulfonamide-substrate-scope.png
   :alt: Image of sulfonamide substrate scope


Many of the substrates appear similar to one another and thereby redundant, but in reality the core sulfone moiety and the use of the same coupling partner when evaluating functional group tolerance accounts for this apparent shortcoming. Also of note is the region of high similarity along the diagonal where the substrates often differ by a single halide heteratom or substitution pattern.

Installing AIMSim
-----------------

It is recommended to install ``AIMSim`` in a virtual environment with `\ ``conda`` <https://docs.conda.io/en/latest/>`_ or Python's `\ ``venv`` <https://docs.python.org/3/library/venv.html>`_.

``pip``
^^^^^^^^^^^

``AIMSim`` can be installed with a single command using Python's package manager ``pip``\ :
``pip install aimsim``
This command also installs the required dependencies.

..

   [!NOTE]
   Looking to use AIMSim for descriptor calculation or extend its functionality? ``AIMSim``\ 's core modules for creating molecules, calculating descriptors, and comparing the results are available without support for plotting or visualization in the PyPI package ``aimsim_core``.


``conda``
^^^^^^^^^^^^^

``AIMSim`` is also available with the ``conda`` package manager via:
``conda install -c conda-forge aimsim``
This will install all dependencies from ``conda-forge``.

Note for mordred-descriptor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

AIMSim v1 provided direct support for the descriptors provided in the ``mordred`` package but unfortunately the original ``mordred`` is now abandonware.
The **unofficial** `\ ``mordredcommunity`` <https://github.com/JacksonBurns/mordred-community>`_ is now used in version 2.1 and newer to deliver the same features but with support for modern Python.

Running AIMSim
--------------

``AIMSim`` is compatible with Python 3.8 to 3.12.
Start ``AIMSim`` with a graphical user interface:

``aimsim``

Start ``AIMSim`` with a prepared configuration YAML file (\ ``config.yaml``\ ):

``aimsim config.yaml``

Currently Implemented Fingerprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Morgan Fingerprint (Equivalent to the ECFP fingerprints)
#. RDKit Topological Fingerprint
#. RDKit Daylight Fingerprint

*The following are available via command line use (config.yaml) only:*


#. MinHash Fingerprint (see `MHFP <https://github.com/reymond-group/mhfp>`_\ )
#. All fingerprints available from the `ccbmlib <https://github.com/vogt-m/ccbmlib>`_ package (\ *specify 'ccbmlib:descriptorname' for command line input*\ ).
#. All descriptors and fingerprints available from `PaDELPy <https://github.com/ecrl/padelpy>`_\ , an interface to PaDEL-Descriptor. (\ *specify 'padelpy:desciptorname' for command line input.*\ ).
#. All descriptors available through the `Mordred <https://github.com/mordred-descriptor/mordred>`_ library (\ *specify 'mordred:desciptorname' for command line input.*\ ). To enable this option, you must install with ``pip install 'aimsim[mordred]'`` (see disclaimer in the Installation section above).

Currently Implemented Similarity Scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

44 commonly used similarity scores are implemented in AIMSim.
Additional L0, L1 and L2 norm based similarities are also implemented. `View our Online Documentation <https://vlachosgroup.github.io/AIMSim/implemented_metrics.html>`_ for a complete list of implemented similarity scores.

Currently Implemented Functionalities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#. Measure Search: Automate the search of fingerprint and similarity metric (called a "measure") using the following algorithm:
   Step 1: Select an arbitrary featurization scheme.
   Step 2: Featurize the molecule set using the selected scheme.
   Step 3: Choose an arbitrary similarity measure.
   Step 4: Select each molecule’s nearest and furthest neighbors in the set using the similarity measure.
   Step 5: Measure the correlation between a molecule’s QoI and its nearest neighbor’s QoI.
   Step 6: Measure the correlation between a molecule’s QoI and its further neighbor’s QoI.
   Step 7: Define a score which maximizes the value in Step 5 and minimizes the value in Step 6.
   Step 8: Iterate Steps 1 – 7 to select the featurization scheme and similarity measure to maximize the result of Step 7. 
#. 
   See Property Variation with Similarity: Visualize the correlation in the QoI between nearest neighbor molecules (most similar pairs in the molecule set) and between the furthest neighbor molecules (most dissimilar pairs in the molecule set). This is used to verify that the chosen measure is appropriate for the task.

#. 
   Visualize Dataset: Visualize the diversity of the molecule set in the form of a pairwise similarity density and a similarity heatmap of the molecule set. Embed the molecule set in 2D space using using principal component analysis (PCA)[3], multi-dimensional scaling[4], t-SNE[5], Spectral Embedding[6], or Isomap[7].

#. 
   Compare Target Molecule to Molecule Set: Run a similarity search of a molecule against a database of molecules (molecule set). This task can be used to identify the most similar (useful in virtual screening operations) or most dissimilar (useful in application that require high diversity such as training set design for machine learning models) molecules.

#. 
   Cluster Data: Cluster the molecule set. The following algorithms are implemented: 

For arbitrary molecular features or similarity metrics with defined Euclidean distances: K-Medoids[3] and Ward[8] (hierarchical clustering).

For binary fingerprints: Complete, single and average linkage hierarchical clustering[8].

The clustered data is plotted in two dimensions using principal component analysis (PCA)[3], multi-dimensional scaling[4], or TSNE[5].


#. Outlier Detection: Using an isolation forest, check for which molecules are potentially novel or are outliers according to the selected descriptor. Output can be directly to the command line by specifiying ``output`` to be ``terminal`` or to a text file by instead providing a filename.

Contributors
------------

Developer: Himaghna Bhattacharjee, Vlachos Research Lab. (\ `LinkedIn <www.linkedin.com/in/himaghna-bhattacharjee>`_\ )

Developer: Jackson Burns, Don Watson Lab. (\ `Personal Site <https://www.jacksonwarnerburns.com/>`_\ )

``AIMSim`` in the Literature
--------------------------------


* `Applications of Artificial Intelligence and Machine Learning Algorithms to Crystallization <https://doi.org/10.1021/acs.chemrev.2c00141>`_
* `Recent Advances in Machine-Learning-Based Chemoinformatics: A Comprehensive Review <https://doi.org/10.3390/ijms241411488>`_

Developer Notes
---------------

Issues and Pull Requests are welcomed! To propose an addition to ``AIMSim`` open an issue and the developers will tag it as an *enhancement* and start discussion.

``AIMSim`` includes an automated testing apparatus operated by Python's *unittest* built-in package. To execute tests related to the core functionality of ``AIMSim``\ , run this command:

``python -m unittest discover``

Full multiprocessing speedup and efficiency tests take more than 10 hours to run due to the number of replicates required. To run these tests, create a file called ``.speedup-test`` in the ``AIMSim`` directory and execute the above command as shown.

To manually build the docs, execute the following with ``sphinx`` and ``m2r`` installed and from the ``/docs`` directory:

``m2r ../README.md | mv ../README.rst . | sphinx-apidoc -f -o . .. | make html | cp _build/html/* .``

Documentation should manually build on push to master branch via an automated GitHub action.

For packaging on PyPI:

``python -m build; twine upload dist/*``

Be sure to bump the version in ``__init__.py``.

Citation
--------

If you use this code for scientific publications, please cite the following paper.

Himaghna Bhattacharjee, Jackson Burns, Dionisios G. Vlachos, AIMSim: An accessible cheminformatics platform for similarity operations on chemicals datasets, Computer Physics Communications, Volume 283, 2023, 108579, ISSN 0010-4655, https://doi.org/10.1016/j.cpc.2022.108579.

License
-------

This code is made available under the terms of the *MIT Open License*\ :

Copyright (c) 2020-2027 Himaghna Bhattacharjee & Jackson Burns

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

Works Cited
-----------

[1] Collins, K. and Glorius, F., A robustness screen for the rapid assessment of chemical reactions. Nature Chem 5, 597–601 (2013). https://doi.org/10.1038/nchem.1669

[2] Chen, Y., Murray, P.R.D., Davies, A.T., and Willis M.C., J. Am. Chem. Soc. 140 (28), 8781-8787 (2018). https://doi.org/10.1021/jacs.8b04532

[3] Hastie, T., Tibshirani R. and Friedman J., The Elements of statistical Learning: Data Mining, Inference, and Prediction, 2nd Ed., Springer Series in Statistics (2009).

[4] Borg, I. and Groenen, P.J.F., Modern Multidimensional Scaling: Theory and Applications, Springer Series in Statistics (2005).

[5] van der Maaten, L.J.P. and Hinton, G.E., Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9:2579-2605 (2008).

[6] Ng, A.Y., Jordan, M.I. and Weiss, Y., On Spectral Clustering: Analysis and an algorithm. ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, MIT Press (2001).

[7] Tenenbaum, J.B., De Silva, V. and Langford, J.C, A global geometric framework for nonlinear dimensionality reduction, Science 290 (5500), 2319-23 (2000). https://doi.org/10.1126/science.290.5500.2319.

[8] Murtagh, F. and Contreras, P., Algorithms for hierarchical clustering: an overview. WIREs Data Mining Knowl Discov (2011). https://doi.org/10.1002/widm.53
