"""

"""
from argparse import ArgumentParser

import yaml

from molSim import task_manager

# import numpy as np
# from scipy.stats import pearsonr
# import yaml

# from molSim.chemical_datastructures import MoleculeSet, Molecule
# from molSim.plotting_scripts import plot_density, plot_heatmap, plot_parity


# def get_molecule_database(database_configs):
#     """Create a MoleculeSet object based on parameters in the config file

#     Parameters
#     ----------
#     database_configs: dict
#         Configurations for initializing the MoleculeSet object.

#     Returns
#     -------
#     molecule_database: MoleculeSet object
#         Molecular database initialized with the parameters.

#     """
#     molecule_database_src = database_configs.get('molecule_database', None)
#     if molecule_database_src is None:
#         raise IOError('<< molecule_database >> field not set in config file')
#     is_verbose = database_configs.get('is_verbose', True)
#     similarity_measure = database_configs.get('similarity_measure',
#                                               'tanimoto_similarity')
#     molecular_descriptor = database_configs.get('molecular_descriptor',
#                                                 'morgan_fingerprint')
#     molecule_database = MoleculeSet(
#                                    molecule_database_src=molecule_database_src,
#                                    similarity_measure=similarity_measure,
#                                    molecular_descriptor=molecular_descriptor,
#                                    is_verbose=is_verbose)
#     return molecule_database


# """ Tasks"""


# def compare_target_molecule(target_molecule,
#                             molecule_set,
#                             out_fpath,
#                             **pdf_plot_kwargs):
#     """
#     Compare a target molecule with molecular database in terms
#     of similarity.
#     Parameters
#     ----------
#     target_molecule: Molecule object
#         Target molecule.
#     molecule_set: MoleculeSet object
#         Database of molecules to compare against.
#     out_fpath: str
#         Filepath to output results. If None, results are not saved and
#         simply displayed to IO.


#     """
#     target_similarity = target_molecule.compare_to_molecule_set(molecule_set)
#     most_similar_mol = molecule_set.molecule_database[
#         np.argmax(target_similarity)]
#     least_similar_mol = molecule_set.molecule_database[
#         np.argmin(target_similarity)]

#     text_prompt = ''
#     text_prompt += f'***** FOR MOLECULE {target_molecule.mol_text} *****\n\n'
#     text_prompt += '****Maximum Similarity Molecule ****\n'
#     text_prompt += f'Molecule: {most_similar_mol.mol_text}\n'
#     text_prompt += 'Similarity: '
#     text_prompt += str(max(target_similarity))
#     text_prompt += '\n'
#     text_prompt += '****Minimum Similarity Molecule ****\n'
#     text_prompt += f'Molecule: {least_similar_mol.mol_text}\n'
#     text_prompt += 'Similarity: '
#     text_prompt += str(min(target_similarity))
#     if out_fpath is None:
#         print(text_prompt)
#     else:
#         if molecule_set.is_verbose:
#             print(text_prompt)
#         print('Writing to file ', out_fpath)
#         with open(out_fpath, "w") as fp:
#             fp.write(text_prompt)
#     plot_density(target_similarity, **pdf_plot_kwargs)


# def visualize_dataset(molecule_database, task_configs):
#     """ Visualize essential properties of the dataset.

#     Parameters
#     ----------
#     molecule_database: MoleculeSet object
#         Molecular database initialized with the parameters.
#     task_configs : dict
#         The parameters needed for the visualizations.

#     Plots Generated
#     ---------------
#     1. Heatmap of Molecular Similarity.
#     2. PDF of the similarity distribution of the molecules in the database.

#     """
#     similarity_matrix = molecule_database.get_similarity_matrix()
#     if molecule_database.is_verbose:
#         print('Plotting similarity heatmap')
#     plot_heatmap(similarity_matrix, **task_configs.get(
#                                             'pairwise_heatmap_settings', {}))
#     if task_configs.get('pairwise similarity', None):
#         if molecule_database.is_verbose:
#             print('Generating pairwise similarities')
#         pairwise_similarity_vector = np.array([similarity_matrix[row, col]
#                                                for row, col
#                                                in zip(
#                                                  range(
#                                                    similarity_matrix.shape[0]),
#                                                  range(
#                                                    similarity_matrix.shape[1]))
#                                                if row < col])
#         if molecule_database.is_verbose:
#             print('Plotting density of pairwise similarities')
#         plot_density(pairwise_similarity_vector,
#                      **task_configs.get('pairwise similarity'))


# def show_property_variation_w_similarity(molecule_database, task_configs):
#     """Plot the variation of molecular property with molecular fingerprint.

#     Parameters
#     ----------
#     molecule_database : Molecules object
#         Molecules object of the molecule database.
#     task_configs : dict
#         The parameters needed for the visualizations.

#     """
#     similar_mol_pairs = molecule_database.get_most_similar_pairs()
#     similarity_plot_params = {
#                                 'xlabel': 'Reference Molecule Property',
#                                 'ylabel': 'Most Similar Molecule Property'
#                               }

#     similarity_plot_params.update(**task_configs.get(
#                                         'similarity_plot_settings', {}))
#     reference_mol_properties, similar_mol_properties = [], []
#     for mol_pair in similar_mol_pairs:
#         mol1_property = mol_pair[0].get_mol_property_val()
#         mol2_property = mol_pair[1].get_mol_property_val()
#         if mol1_property and mol2_property:
#             reference_mol_properties.append(mol1_property)
#             similar_mol_properties.append(mol2_property)
#     if molecule_database.is_verbose:
#         print('Plotting Responses of Similar Molecules')
#     plot_parity(reference_mol_properties,
#                 similar_mol_properties,
#                 **similarity_plot_params)
#     pearson_coff_of_responses = pearsonr(reference_mol_properties,
#                                          similar_mol_properties)
#     print(f'Pearson Correlation in the properties of the '
#           f'most similar molecules is: {pearson_coff_of_responses[0]}   '
#           f'2 tailed p-value: {pearson_coff_of_responses[1]}')

#     if task_configs.pop('get most dissimilar', False):
#         dissimilar_mol_pairs = molecule_database.get_most_dissimilar_pairs()
#         dissimilarity_plot_params = {
#             'xlabel': 'Reference Molecule Property',
#             'ylabel': 'Most Dissimilar Molecule Property'
#         }
#         dissimilarity_plot_params.update(**task_configs.get(
#             'dissimilarity plot parameters', None))
#         reference_mol_properties, dissimilar_mol_properties = [], []
#         for mol_pair in dissimilar_mol_pairs:
#             mol1_property = mol_pair[0].get_mol_property_val()
#             mol2_property = mol_pair[1].get_mol_property_val()
#             if mol1_property and mol2_property:
#                 reference_mol_properties.append(mol1_property)
#                 dissimilar_mol_pairs.append(mol2_property)
#         if molecule_database.is_verbose:
#             print('Plotting Responses of Dissimilar Molecules')
#         plot_parity(reference_mol_properties,
#                     dissimilar_mol_properties,
#                     **dissimilarity_plot_params)
#         pearson_coff_of_dissimilar_responses = pearsonr(reference_mol_properties,
#                                              dissimilar_mol_properties)
#         print(f'Pearson Correlation in the properties of the '
#               f'most dissimilar molecules is: '
#               f'{pearson_coff_of_dissimilar_responses[0]}   '
#               f'2 tailed p-value: {pearson_coff_of_dissimilar_responses[1]}')


# def launch_tasks(molecule_database, tasks):
#     """Sequentially launches all the tasks from the configuration file.

#     Parameters
#     ----------
#     molecule_database: MoleculeSet object
#         Molecular database initialized with the parameters.
#     tasks : dict
#         The tasks field of the config yaml containing various tasks
#         and their parameters

#     """
#     for task, task_configs in tasks.items():
#         if task == 'compare_target_molecule':
#             target_molecule_smiles = task_configs.get('target_molecule_smiles')
#             target_molecule_src = task_configs.get('target_molecule_src')
#             if target_molecule_smiles:
#                 target_molecule = Molecule(mol_smiles=target_molecule_smiles)
#             elif target_molecule_src:
#                 target_molecule = Molecule(mol_src=target_molecule_src)
#             else:
#                 raise IOError('Target molecule source is not specified '
#                               f'for task {task}')
#             save_to_file = task_configs.get('save_to_file', None)
#             pdf_plot_kwargs = task_configs.get('plot_settings')
#             compare_target_molecule(target_molecule=target_molecule,
#                                     molecule_set=molecule_database,
#                                     out_fpath=save_to_file,
#                                     **pdf_plot_kwargs)
#         elif task == 'visualize_dataset':
#             visualize_dataset(molecule_database, task_configs)
#         elif task == 'show_property_variation_w_similarity':
#             show_property_variation_w_similarity(molecule_database,
#                                                  task_configs)
#         else:
#             raise NotImplementedError(
#                 f'{task} entered in the <<task>> field is not implemented')

#     input("Press enter to terminate (plots will be closed).")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config yaml file.')
    args = parser.parse_args()
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    tasks = configs.pop('tasks', None)
    if tasks is None:
        raise IOError('<< tasks >> field not set in config file')
    task_manager.launch_tasks(molecule_database_configs=configs,
                              tasks=tasks)
