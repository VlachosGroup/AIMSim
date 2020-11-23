"""Visualize similarity and diversity in chemical structure.
@author: Himaghna Bhattacharjee, Vlachos Research Lab.

Notes
-----
This script can be run as:
>> python molecular_similarity.py config.yaml

config.yaml is the configuration file that is needed to choose tasks and
their settings.

"""
from argparse import ArgumentParser
from glob import glob
import os.path
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem import AllChem
from seaborn import heatmap, kdeplot
import yaml








# Some plotting methods
def plot_density(similarity_vector, title, color, shade, **kwargs):
    """Plot the similarity density

    Attributes
    ----------
    similarity_vector : list or numpy ndarray
        Vector of similarity scores to be plotted.
    title : str
        Plot title
    color : str
        Color of the plot.
    shade : bool
        To shade the plot or not.
    kwargs : dict
        Keyword arguments to modify plot. Some common ones:
        bw : Thickness. Default 0.01.

    """
    # get params
    bw = float(kwargs.get('bw', 0.01))
    plt.figure()
    plt.rcParams['svg.fonttype'] = 'none'
    kdeplot(similarity_vector, shade=shade, color=color, bw=bw)
    plt.xlabel('Samples', fontsize=20)
    plt.ylabel('Similarity Density', fontsize=20)
    if title is not None:
        plt.title(title, fontsize=20)
    plt.show(block=False)


def show_property_variation_w_similarity(config, molecules, isVerbose):
    """Plot the variation of molecular property with molecular fingerprint.

    Parameters
    ----------
    config : dict
        Configuration file.
    molecules : Molecules object
        Molecules object of the molecule database.

    """
    # load properties
    try:
        mol_prop_file = config['property_file']
    except KeyError as e:
        raise e('Property file not provided.')
    with open(mol_prop_file, "r") as fp:
        data_lines = fp.readlines()
    names, properties = [], []
    for line in data_lines:
        name_, prop = line.split()   # each line is <<name property>>
        names.append(name_)
        properties.append(prop)
    for mol in molecules.mols:
        # find corresponding molecule in the property file, assign property
        if isVerbose:
            print(f'Assigning property of {mol.name_}')
        mol_id = names.index(mol.name_)
        mol.mol_property = float(properties[mol_id])
    if config.get('most_dissimilar', False):
        mol_pairs = molecules.get_most_dissimilar_pairs()  # (mol1, mol2)
    else:
        mol_pairs = molecules.get_most_similar_pairs()  # (mol1, mol2)
    property_mols1, property_mols2 = [], []
    for mol_pair in mol_pairs:
        if mol_pair[0] == mol_pair[1]:
            if isVerbose:
                print(mol_pair)
        # property of first molecule in pair. Discard if property not set.
        property_mol1 = molecules.mols[mol_pair[0]].mol_property
        if property_mol1 is None:
            continue
        # property of second molecules in pair. Discard if property not set.
        property_mol2 = molecules.mols[mol_pair[1]].mol_property
        if property_mol2 is None:
            continue
        property_mols1.append(property_mol1)
        property_mols2.append(property_mol2)

    def plot_parity(x, y, **kwargs):
        """Plot parity plot of x vs y.

        Parameters
        ----------
        x: n x 1 numpy array: values plotted along x axis
        y: n x 1 numpy array: values plotted along y axis

        Returns
        -------
        if kwargs.show_plot set to False, returns pyplot axis.

        """
        plot_params = {
            'alpha': 0.7,
            's': 10,
            'c': 'green',
        }
        if kwargs is not None:
            plot_params.update(kwargs)
            plt.figure()
        plt.rcParams['svg.fonttype'] = 'none'
        plt.scatter(
            x=x, y=y, alpha=plot_params['alpha'], s=plot_params['s'],
            c=plot_params['c'])
        max_entry = max(max(x), max(y)) + plot_params.get('offset', 5.0)
        min_entry = min(min(x), min(y)) - plot_params.get('offset', 5.0)
        axes = plt.gca()
        axes.set_xlim([min_entry, max_entry])
        axes.set_ylim([min_entry, max_entry])
        plt.plot(
            [min_entry, max_entry],
            [min_entry, max_entry],
            color=plot_params.get('linecolor', 'black'))
        plt.title(
            plot_params.get('title', ''),
            fontsize=plot_params.get('title_fontsize', 24))
        plt.xlabel(
            plot_params.get('xlabel', ''),
            fontsize=plot_params.get('xlabel_fontsize', 20))
        plt.ylabel(
            plot_params.get('ylabel', ''),
            fontsize=plot_params.get('ylabel_fontsize', 20))
        plt.xticks(fontsize=plot_params.get('xticksize', 24))
        plt.yticks(fontsize=plot_params.get('yticksize', 24))
        start, end = axes.get_xlim()
        stepsize = (end - start) / 5
        axes.xaxis.set_ticks(np.arange(start, end, stepsize))
        axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        # set y tick stepsize
        start, end = axes.get_ylim()
        stepsize = (end - start) / 5
        axes.yaxis.set_ticks(np.arange(start, end, stepsize))
        axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        plt.show(block=False)

    plot_parity(
        property_mols1, property_mols2, xlabel='Response Molecule 1',
        ylabel='Response Molecule 2')


def compare_target_molecule(config, db_molecules):
    """Compare a target molecule with molecular database in terms
    of similarity.

    Parameters
    ----------
    config : dict
        Configurations as key value pairs.
    db_molecules : Molecules
        Molecules object representing the database.

    """
    # get sub-tasks
    try:
        target_mol_config = config['target_molecule']
    except KeyError as e:
        raise e('No target molecule specified')
    show_pdf_configs = config.get('show_pdf', False)
    identify_closest_furthest = config.get('identify_closest_furthest', False)

    if os.path.isfile(target_mol_config):
        target_fname, target_ext = \
            os.path.basename(target_mol_config).split('.')
        if target_ext == 'pdb':
            # read pdb file
            target_mol_object = Chem.MolFromPDBFile(target_mol_config)
            # for a pdb file, name is the filename
            target_name = target_fname
        elif target_ext == 'txt':
            # read smile from txt file
            with open(target_mol_config, "r") as fp:
                data = fp.readline()
            smile = data.split[0]  # assume first word is smile
            target_mol_object = Chem.MolFromSmiles(smile)
            # for a smile string, name is the smile string
            target_name = smile
    else:
        # assume target_mol is a smile string
        target_mol_object = Chem.MolFromSmiles(target_mol_config)
        # supplied smile string is also target name
        target_name = target_mol_config
    if target_mol_object is None:
        raise IOError('Target Molecules could not be loaded')
    # sanitize
    rdmolops.Kekulize(target_mol_object)
    target_molecule = Molecule(target_mol_object, name_=target_name)
    target_similarity = [
        target_molecule.get_similarity_to_molecule(
            ref_mol, similarity_measure=db_molecules.similarity_measure,
            molecular_descriptor=db_molecules.molecular_descriptor)
        for ref_mol in db_molecules.mols if ref_mol.name_ != target_name]

    def output_max_min_similarity(out_fpath):
        """Finds the most and least similar molecule to the target molecule
        and outputs their name and similarity score to an output file.

        Parameters
        ----------
        out_fpath : str
            Complete filepath of the output file to be generated.

        """
        with open(out_fpath, "w") as fp:
            fp.write(f'***** FOR MOLECULE {target_molecule.name_} *****\n\n')
            fp.write('****Maximum Similarity Molecule ****\n')
            fp.write('Molecule: ')
            fp.write(
                str(db_molecules.mols[np.argmax(target_similarity)].name_))
            fp.write('\n')
            fp.write('Similarity: ')
            fp.write(
                str(target_similarity[np.argmax(target_similarity)]))
            fp.write('\n')
            fp.write('****Minimum Similarity Molecule ****\n')
            fp.write('Molecule: ')
            fp.write(
                str(db_molecules.mols[np.argmin(target_similarity)].name_))
            fp.write('\n')
            fp.write('Similarity: ')
            fp.write(
                str(target_similarity[np.argmin(target_similarity)]))

    if show_pdf_configs is not None:
        pdf_color = show_pdf_configs.get('pdf_color', 'violet')
        pdf_shade = show_pdf_configs.get('pdf_shade', True)
        pdf_title = show_pdf_configs.get('pdf_title', None)
        plot_density(
            target_similarity, title=pdf_title,
            color=pdf_color, shade=pdf_shade)
    if identify_closest_furthest:
        output_max_min_similarity(
            out_fpath=identify_closest_furthest['out_file_path'])


def visualize_dataset(config, db_molecules):

    def draw_similarity_heatmap(**kwargs):
        """Plot a heatmap of the distance matrix.

        """
        # load sub-tasks
        xticklabels = kwargs.get('xticklabels', False)
        yticklabels = kwargs.get('yticklabels', False)
        cmap = kwargs.get('cmap', 'autumn')
        mask_upper = kwargs.get('mask_upper', True)
        annotate = kwargs.get('annotate', False)

        if db_molecules.similarity_matrix is None:
            db_molecules.generate_similarity_matrix()
        # plot
        plt.figure()
        plt.rcParams['svg.fonttype'] = 'none'
        mask = None
        if mask_upper is True:
            mask = np.triu(np.ones(
                shape=db_molecules.similarity_matrix.shape), k=0)
        heatmap_obj = heatmap(
            db_molecules.similarity_matrix, xticklabels=xticklabels,
            yticklabels=yticklabels, cmap=cmap, mask=mask, annot=annotate)
        plt.show(block=False)
        return heatmap_obj

    def show_db_pdf(**kwargs):
        """Show the probability density distribution of the
        molecular database.

        """
        pdf_color = kwargs.get('pdf_color', 'violet')
        pdf_shade = kwargs.get('pdf_shade', True)
        pdf_title = kwargs.get('pdf_title', None)
        if db_molecules.similarity_matrix is None:
            db_molecules.generate_similarity_matrix()
        lower_diag_indices = np.tril_indices(
            db_molecules.similarity_matrix.shape[0], -1)
        similarity_vector = db_molecules.similarity_matrix[lower_diag_indices]
        plot_density(
            similarity_vector, color=pdf_color, shade=pdf_shade,
            title=pdf_title)

    show_pdf_configs = config.get('show_pdf', False)
    if show_pdf_configs:
        show_db_pdf(**show_pdf_configs)

    show_heatmap_configs = config.get('show_heatmap', False)
    if show_heatmap_configs:
        draw_similarity_heatmap(**show_heatmap_configs)


def sort_tasks(configs):
    """Activate various functionalities based on the 'tasks' field of configs.

    Parameters
    ----------
    configs : dict
        Loaded configuration setting from yaml file.

    """
    # load common parameters
    try:
        tasks = configs['tasks']
    except KeyError:
        raise IOError('<< tasks >> field not set in config file')
    try:
        molecule_database = configs['molecule_database']
    except KeyError:
        raise IOError('<< molecule_database >> field not set in config file')
    verbose = configs.get(
        'verbose')
    similarity_measure = configs.get(
        'similarity_measure', 'tanimoto_similarity')
    molecular_descriptor = configs.get(
        'molecular_descriptor', 'morgan_fingerprint')
    molecules = Molecules(
        mols_src=molecule_database,
        similarity_measure=similarity_measure,
        molecular_descriptor=molecular_descriptor,
        isVerbose=verbose)

    for task, task_configs in tasks.items():
        if task == 'compare_target_molecule':
            compare_target_molecule(task_configs, molecules)
        elif task == 'visualize_dataset':
            visualize_dataset(task_configs, molecules)
        elif task == 'show_property_variation_w_similarity':
            show_property_variation_w_similarity(
                task_configs, molecules, verbose)
        else:
            raise NotImplementedError(
                f'{task} entered in the <<task>> field is not implemented')

    input("Press enter to terminate (plots will be closed).")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config yaml file.')
    args = parser.parse_args()
    # load configs
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    sort_tasks(configs)
