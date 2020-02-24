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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem import AllChem
from seaborn import heatmap, kdeplot
import yaml


class Molecule:
    """Molecular object defined from rdkit mol object.

    """
    def __init__(self, mol, name_=None, mol_property_val=None):
        """Constructor

        Parameters
        ----------
        mol: RDkit mol object
            molecule to be abstracted in the Molecule class.
        name_: str
            Name of the molecule. Default is None.
        mol_property_val: float
            Some property associated with the molecule. This is typically the
            response being studied. E.g. Boiling point, Selectivity etc.
            Default is None.

        """
        self.mol = mol
        self.name_ = name_
        self.mol_property = mol_property_val

    def get_molecular_descriptor(self, molecular_descriptor):
        """Expose a suitable method based on molecular_descriptor

        """
        if molecular_descriptor == 'morgan_fingerprint':
            return self._get_morgan_fingerprint()
        elif molecular_descriptor == 'rdkit_topological':
            return self._get_rdkit_topological_fingerprint()

    def get_similarity(self, similarity_measure, mol1_descrptr, mol2_descrptr):
        """Expose a suitable method based on similarity_measure

        """
        if similarity_measure == 'tanimoto_similarity':
            return DataStructs.TanimotoSimilarity(mol1_descrptr, mol2_descrptr)
        elif similarity_measure == 'neg_l1':
            return -np.linalg.norm(np.asarray(mol1_descrptr)
                - np.asarray(mol2_descrptr), ord=1)
        elif similarity_measure == 'neg_l2':
            return -np.linalg.norm(np.asarray(mol1_descrptr)
                - np.asarray(mol2_descrptr), ord=2)

    def _get_morgan_fingerprint(self, radius=3, n_bits=None):
        """Generate a morgan fingerprint.

        Parameters
        ----------
        radius: int: radius of fingerprint, 3 corresponds to diameter 6.
                    Default 3.
        n_bits: int: Number of bits to use if Morgan Fingerprint wanted as
            a bit vector. If set to None, Morgan fingerprint returned
            as count. Default is None.

        Returns
        -------
        morgan_fingerprint: int
        """
        if n_bits is None:
            return AllChem.GetMorganFingerprint(self.mol, radius)
        else:
            return AllChem.GetMorganFingerprintAsBitVect(self.mol, radius,
                nBits=n_bits)

    def _get_rdkit_topological_fingerprint(self, min_path=1, max_path=7):
        return rdmolops.RDKFingerprint(
            self.mol, minPath=min_path, maxPath=max_path)

    def get_similarity_to_molecule(
            self, target_mol, similarity_measure='tanimoto',
            molecular_descriptor='morgan_fingerprint'):
        """Get a similarity metric to a target molecule

        Parameters
        ----------
        target_mol: Molecule object: Target molecule.
            Similarity score is with respect to this molecule
        similarity_measure: str
            The similarity metric used.
            *** Supported Metrics ***
            'tanimoto': Jaccard Coefficient/ Tanimoto Similarity
                    0 (not similar at all) to 1 (identical)
            'neg_l1': Negative L1 norm of |x1 - x2|
            'neg_l2': Negative L2 norm of |x1 - x2|
        molecular_descriptor : str
            The molecular descriptor used to encode molecules.
            *** Supported Descriptors ***
            'morgan_fingerprint'

        Returns
        -------
        similarity_score: float
            Similarity coefficient by the chosen method.

        """

        similarity_score = self.get_similarity(
            similarity_measure,
            self.get_molecular_descriptor(molecular_descriptor),
                target_mol.get_molecular_descriptor(molecular_descriptor))

        return similarity_score


class Molecules:
    """Collection of Molecule objects.

    Attributes
    ----------
    mols: List
        List of Molecule objects.
    similarity_measure : str
        Similarity measure used.
    molecular_descriptor : str
        Molecular descriptor used. Currently implements
            - 'morgan_fingerprint'
            - 'rdkit_topological'
    similarity_matrix: numpy ndarray
        n_mols X n_mols numpy matrix of pairwise similarity scores.

    Methods
    -------
    generate_similarity_matrix()
        Set the similarity_matrix attribute.
    get_most_similar_pairs()
        Get the indexes of the most similar molecules as tuples.

    """
    def __init__(self, mols_src, similarity_measure, molecular_descriptor):
        self.similarity_measure = similarity_measure
        self.molecular_descriptor = molecular_descriptor
        self.mols = self._set_mols(mols_src)
        self.similarity_matrix = None

    def _set_mols(self, mols_src):
        """Return list of Molecule objects from mols_src.

        """
        mol_list = []
        if os.path.isdir(mols_src):
            print(f'Searching for *.pdb files in {mols_src}')
            for molfile in glob(os.path.join(mols_src, '*.pdb')):
                mol_object = Chem.MolFromPDBFile(molfile)
                mol_name = os.path.basename(molfile).replace('.pdb', '')
                if mol_object is None:
                    print(f'{molfile} could not be imported. Skipping')
                    continue
                rdmolops.Kekulize(mol_object)
                mol_list.append(Molecule(mol_object, mol_name))
        elif os.path.isfile(mols_src):
            print(f'Reading SMILES strings from {mols_src}')
            with open(mols_src, "r") as fp:
                smiles_data = fp.readlines()
            for count, line in enumerate(smiles_data):
                # Assumes that the first column contains the smiles string
                smile = line.split()[0]
                print(f'Processing {smile} ({count + 1}/{len(smiles_data)})')
                mol_object = Chem.MolFromSmiles(smile)
                if mol_object is None:
                    print(f'{smile} could not be loaded')
                    continue
                # sanitize
                rdmolops.Kekulize(mol_object)
                mol_name = smile
                mol_list.append(Molecule(mol_object, mol_name))
        else:
            raise FileNotFoundError(
            f'{mols_src} could not be found.Please enter valid ' \
                'foldername or path of a text file w/ SMILES strings')
        if len(mol_list) == 0:
            raise UserWarning('No molecular files found in the location!')
        return mol_list

    def generate_similarity_matrix(self):
        n_mols = len(self.mols)
        self.similarity_matrix = np.zeros(shape=(n_mols, n_mols))
        for id, mol in enumerate(self.mols):
            for target_id in range(id, n_mols):
                print(f'checking molecules num {target_id+1} against {id+1}')
                self.similarity_matrix[id, target_id] = \
                    mol.get_similarity_to_molecule(
                        self.mols[target_id],
                        similarity_measure=self.similarity_measure,
                        molecular_descriptor=self.molecular_descriptor)
                # symmetric matrix entry
                self.similarity_matrix[target_id, id] = \
                    self.similarity_matrix[id, target_id]

    def get_most_similar_pairs(self):
        """Get pairs of samples which are most similar.

        Returns
        -------
        List(Tuple(int, int))
            List of pairs of indices closest to one another.

        """
        # If not set, set similarity_matrix.
        if self.similarity_matrix is None:
            self.generate_similarity_matrix()

        n_samples = self.similarity_matrix.shape[0]
        found_samples = [0 for _ in range(n_samples)]
        out_list = []
        for index, row in enumerate(self.similarity_matrix):
            if found_samples[index]:
                # if  species has been identified before
                continue
            post_diag_closest_index = np.argmax(row[(index +1):]) \
                + index + 1 if index < n_samples-1 else -1
            pre_diag_closest_index = np.argmax(row[:index]) if index > 0 \
                else -1
            # if either (pre_) post_diag_closest_index not set, the
            # closest_index_index is set to the (post_) pre_diag_closest_index
            if pre_diag_closest_index == -1:
                closest_index_index = post_diag_closest_index
            if post_diag_closest_index == -1:
                closest_index_index = pre_diag_closest_index
            # if both pre and post index set, closest_index_index set to index
            # with min distance. In case of tie, post_diag_closest_index set
            else:
                # choose the index which has max correlation
                closest_index_index = ( post_diag_closest_index if
                    row[post_diag_closest_index] >= row[pre_diag_closest_index]
                        else pre_diag_closest_index )
            out_list.append((index, closest_index_index))
            # update list
            found_samples[closest_index_index] = 1
            found_samples[index] = 1
        return out_list

    def get_most_dissimilar_pairs(self):
        """Get pairs of samples which are least similar.

        Returns
        -------
        List(Tuple(int, int))
            List of pairs of indices closest to one another.

        """
        # If not set, set similarity_matrix.
        if self.similarity_matrix is None:
            self.generate_similarity_matrix()

        n_samples = self.similarity_matrix.shape[0]
        found_samples = [0 for _ in range(n_samples)]
        out_list = []
        for index, row in enumerate(self.similarity_matrix):
            if found_samples[index]:
                # if  species has been identified before
                continue
            furthest_index = np.argmin(row)
            out_list.append((index, furthest_index))
            # update list
            found_samples[furthest_index] = 1
            found_samples[index] = 1
        return out_list


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
    plt.rcParams['svg.fonttype'] = 'none'
    kdeplot(similarity_vector, shade=shade, color=color, bw=bw)
    plt.xlabel('Samples', fontsize=20)
    plt.ylabel('Similarity Density', fontsize=20)
    if title is not None:
        plt.title(title, fontsize=20)
    plt.show()


def show_property_variation_w_similarity(config, molecules):
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
        plt.rcParams['svg.fonttype'] = 'none'
        plt.scatter(
            x=x, y=y, alpha=plot_params['alpha'], s=plot_params['s'],
            c=plot_params['c'])
        max_entry = max(max(x), max(y)) + plot_params.get('offset', 5.0)
        min_entry = min(min(x), min(y))  - plot_params.get('offset', 5.0)
        axes = plt.gca()
        axes.set_xlim([min_entry, max_entry])
        axes.set_ylim([min_entry, max_entry])
        plt.plot([min_entry, max_entry], [min_entry, max_entry],
                color=plot_params.get('linecolor', 'black'))
        plt.title(
            plot_params.get('title', ''),
            fontsize=plot_params.get('title_fontsize', 24))
        plt.xlabel(plot_params.get('xlabel', ''),
                    fontsize=plot_params.get('xlabel_fontsize', 20))
        plt.ylabel(plot_params.get('ylabel', ''),
                    fontsize=plot_params.get('ylabel_fontsize', 20))
        plt.xticks(fontsize=plot_params.get('xticksize',24))
        plt.yticks(fontsize=plot_params.get('yticksize',24))
        start, end = axes.get_xlim()
        stepsize = (end - start) / 5
        axes.xaxis.set_ticks(np.arange(start, end, stepsize))
        axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        # set y tick stepsize
        start, end = axes.get_ylim()
        stepsize = (end - start) / 5
        axes.yaxis.set_ticks(np.arange(start, end, stepsize))
        axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        plt.show()

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
        plt.rcParams['svg.fonttype'] = 'none'
        mask = None
        if mask_upper is True:
            mask = np.triu(np.ones(
                shape=db_molecules.similarity_matrix.shape), k=0)
        heatmap_obj = heatmap(
            db_molecules.similarity_matrix, xticklabels=xticklabels,
            yticklabels=yticklabels, cmap=cmap, mask=mask, annot=annotate)
        plt.show()

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
    similarity_measure = configs.get(
        'similarity_measure', 'tanimoto_similarity')
    molecular_descriptor = configs.get(
        'molecular_descriptor', 'morgan_fingerprint')
    molecules = Molecules(
        mols_src=molecule_database,
        similarity_measure=similarity_measure,
        molecular_descriptor=molecular_descriptor)

    for task, task_configs in tasks.items():
        if task == 'compare_target_molecule':
            compare_target_molecule(task_configs, molecules)
        elif task == 'visualize_dataset':
            visualize_dataset(task_configs, molecules)
        elif task == 'show_property_variation_w_similarity':
            show_property_variation_w_similarity(task_configs, molecules)
        else:
            raise NotImplementedError(
            f'{task} entered in the <<task>> field is not implemented')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config yaml file.')
    args = parser.parse_args()
    # load configs
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    sort_tasks(configs)
