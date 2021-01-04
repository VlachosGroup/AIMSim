molSim is a tool for visualizing diversity in your molecular data-set using graph theory. 

<b>Why Do We Need To Visualize Molecular Similarity / Diversity?</b>

There are two broad contexts where it is helpful to visualize the diversity of a molecular dataset:

<i> Experimental Synthesis </i>

For a chemist, synthesizing new molecules with targeted properties is often a laborious and time consuming task.
In such a case, it becomes useful to check the similarity of a newly proposed (un-synthesized) molecule to the ones already synthesized.
If the proposed molecule is too similar to the existing repertoire of molecules, it will probably not yield not enough new information /
property and thus need not be synthesized. On the other hand, if the aim is to replicate the properties of a high performing molecule,
it is useful to ensure that each new proposed molecule is similar to the high performing one. In both cases, a chemist can avoid spending
time and effort synthesizing molecules not useful for the project.

<i> Machine Learning Molecular Properties </i>

In the context of machine learning, visualizing the diversity of the training set gives a good idea about its information quality.
A more diverse training data-set yields a more robust model, which generalizes well to unseen data. Additionally, such a visualization can 
identify "clusters of similarity" indicating the need for separately trained models for each cluster.

<b> Dependencies </b>

Use the following command with conda to create an environment:
> conda create --name <env> --file spec-file.txt

1. Python 3+
2. Matplotlib
3. Numpy
4. RDKIT
5. SEABORN
6. PyYAML
7. Pandas 1.0.1+
8. openpyxl

<b> Example Run </b>
>> python -m molSim config.yaml
Tests:
>> python -m unittest discover

<b> Notes </b>

<i> General Workflow </i>

Molecular Structure Information (SMILES strings, *.pdb files etc.) --> Generate a Molecular Graph / Environment Fingerprint
--> Calculate a "similarity score" between moelcules based on some distance between their fingerprints.

<i> Currently Implemented Fingerprints </i>

1. Morgan Fingerprint (Equivalent to the ECFP-6)
2. RDKIT Topological Fingerprint

<i> Currently Implemented Similarity Scores </i>

1. Tanomito Similarity (0 for completely dissimilaar and 1 for identical molecules)

<i> Currently Implemented Functionalities </i>

1. compare_target_molecule: Compare a proposed molecules to existing molecular database. The outputs are a similarity density plot
and/ or the least similar and most similar molecules in the database (to the proposed molecule)

2. visualize_dataset: Visualize the diversity of molecules in existing database. The outputs are a heatmap of similarity scores and/or
a density plot of similarity scores and /or a parity plot showing some molecular property (e.g. boiling point) between 
pairs of most similar molecules. The last output requires the input of the molecular property for each molecule.
This can be inputted as a .txt file containing rows of name property pairs. An example of such a file with fictitious properties is
provided in the file smiles_responses.txt. This option is typically used to check the suitability of the fingerprint / similarity measure
for a property of interest. If they do a good job for the particular property then the parity plot should be scattered around the diagonal.

<b> Credits and Licensing</b>

Lead Developer: Himaghna Bhattacharjee, Vlachos Research Lab. (www.linkedin.com/in/himaghna-bhattacharjee)

Developer: Jackson Burns, Don Watson Lab. ([Personal Site](https://www.jacksonwarnerburns.com/))

License: MIT Open
