"""

"""
from argparse import ArgumentParser

import numpy as np
import yaml

from chemical_datastructures import MoleculeSet, Molecule
from plotting_scripts import plot_density


def get_molecule_database(database_configs):
    """Create a MoleculeSet object based on parameters in the config file

    Parameters
    ----------
    database_configs: dict
        Configurations for initializing the MoleculeSet object.

    Returns
    -------
    molecule_database: MoleculeSet object
        Molecular database initialized with the parameters.

    """
    molecule_database_src = database_configs.get('molecule_database_src', None)
    if molecule_database_src is None:
        raise IOError('<< molecule_database >> field not set in config file')
    is_verbose = database_configs.get('is_verbose', True)
    similarity_measure = database_configs.get('similarity_measure',
                                              'tanimoto_similarity')
    molecular_descriptor = database_configs.get('molecular_descriptor',
                                                'morgan_fingerprint')
    molecule_database = MoleculeSet(
                                   molecule_database_src=molecule_database_src,
                                   similarity_measure=similarity_measure,
                                   molecular_descriptor=molecular_descriptor,
                                   is_verbose=is_verbose)
    return molecule_database


""" Tasks"""


def compare_target_molecule(target_molecule_src,
                            molecule_set,
                            out_fpath,
                            **pdf_plot_kwargs):
    """
    Compare a target molecule with molecular database in terms
    of similarity.
    Parameters
    ----------
    target_molecule_src: str
        Filepath for loading target molecule.
    molecule_set: MoleculeSet object
        Database of molecules to compare against.
    out_fpath: str
        Filepath to output results. If None, results are not saved and
        simply displated to IO.

    Returns
    -------

    """
    target_molecule = Molecule(mol_src=target_molecule_src)
    target_similarity = target_molecule.compare_to_molecule_set(molecule_set)
    most_similar_mol = molecule_set.molecule_database[
        np.argmax(target_similarity)]
    least_similar_mol = molecule_set.molecule_database[
        np.argmin(target_similarity)]

    text_prompt = ''
    text_prompt += f'***** FOR MOLECULE {target_molecule.mol_text} *****\n\n'
    text_prompt += '****Maximum Similarity Molecule ****\n'
    text_prompt += f'Molecule: {most_similar_mol.mol_text}\n'
    text_prompt += 'Similarity: '
    text_prompt += str(max(target_similarity))
    text_prompt += '\n'
    text_prompt += '****Minimum Similarity Molecule ****\n'
    text_prompt += f'Molecule: {least_similar_mol}\n'
    text_prompt += 'Similarity: '
    text_prompt += str(min(target_similarity))
    if out_fpath is None:
        print(text_prompt)
    else:
        if molecule_set.is_verbose:
            print(text_prompt)
        print('Writing to file ', out_fpath)
        with open(out_fpath, "w") as fp:
            fp.write(text_prompt)

    plot_density(target_similarity, **pdf_plot_kwargs)


def launch_tasks(molecule_database, tasks):
    """Sequentially launches all the tasks from the configuration file.

    Parameters
    ----------
    molecule_database: MoleculeSet object
        Molecular database initialized with the parameters.
    tasks : dict
        The tasks field of the config yaml containing various tasks
        and their parameters

    """
    for task, task_configs in tasks.items():
        if task == 'compare_target_molecule':
            target_molecule_src = task_configs.get('target_molecule_src')
            save_to_file = task_configs.get('save_to_file', None)
            pdf_plot_kwargs = task_configs.get('plot_settings')
            compare_target_molecule(target_molecule_src,
                                    molecule_database,
                                    out_fpath=save_to_file,
                                    pdf_plot_kwargs=pdf_plot_kwargs)
        elif task == 'visualize_dataset':
            visualize_dataset(task_configs, molecule_database)
        elif task == 'show_property_variation_w_similarity':
            show_property_variation_w_similarity(
                task_configs, molecule_database, verbose)
        else:
            raise NotImplementedError(
                f'{task} entered in the <<task>> field is not implemented')

    input("Press enter to terminate (plots will be closed).")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config yaml file.')
    args = parser.parse_args()
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    tasks = configs.pop('tasks', None)
    if tasks is None:
        raise IOError('<< tasks >> field not set in config file')
    molecule_database = get_molecule_database(database_configs=configs)
    launch_tasks(molecule_database=molecule_database,
                 tasks=tasks)
