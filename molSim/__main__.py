# -*- coding: utf-8 -*-
"""
molSim is a tool for visualizing diversity in your molecular data-set using
graph theory.
Why Do We Need To Visualize Molecular Similarity / Diversity?
There are two broad contexts where it is helpful to visualize the diversity
of a molecular dataset:

Experimental Synthesis
----------------------
For a chemist, synthesizing new molecules with targeted properties is often
a laborious and time consuming task. In such a case, it becomes useful to
check the similarity of a newly proposed (un-synthesized) molecule to the
ones already synthesized. If the proposed molecule is too similar to the
existing repertoire of molecules, it will probably not yield not enough
new information / property and thus need not be synthesized.
On the other hand, if the aim is to replicate the properties
of a high performing molecule, it is useful to ensure that each
new proposed molecule is similar to the high performing one.
In both cases, a chemist can avoid spending time and effort synthesizing
molecules not useful for the project.

Machine Learning Molecular Properties
-------------------------------------
In the context of machine learning, visualizing the diversity of the
training set gives a good idea about its information quality.
A more diverse training data-set yields a more robust model, which
generalizes well to unseen data. Additionally, such a visualization can
identify "clusters of similarity" indicating the need for separately
trained models for each cluster.

Dependencies
------------

Use the following command with conda to create an environment:

    conda create --name --file spec-file.txt

    Python 3+
    Matplotlib
    Numpy
    RDKIT
    SEABORN
    PyYAML

Example Run
-----------

python molecular_similarity.py config.yaml

Credits and Licensing
---------------------

Lead Developer: Himaghna Bhattacharjee, Vlachos Research Lab.
Developer: Jackson Burns, Don Watson Lab.

License: MIT Open

"""

from argparse import ArgumentParser

import yaml

from molSim import task_manager

parser = ArgumentParser()
parser.add_argument('config', help='Path to config yaml file.')
args = parser.parse_args()
configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

tasks = configs.pop('tasks', None)
if tasks is None:
    raise IOError('<< tasks >> field not set in config file')

task_manager.launch_tasks(molecule_database_configs=configs, tasks=tasks)