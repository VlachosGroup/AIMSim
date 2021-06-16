from abc import ABC, abstractmethod
from copy import deepcopy
from os import makedirs
from os.path import basename

import numpy as np
from scipy.stats import pearsonr

from molSim.chemical_datastructures import MoleculeSet, Molecule
from molSim.plotting_scripts import plot_density, plot_heatmap, plot_parity


class TaskManager:
    def __init__(self, tasks):
        """Sequentially launches all the tasks from the configuration file.

        Parameters
        ----------
        tasks: dict
            The tasks field of the config yaml containing various tasks
            and their parameters.

        """
        self.to_do = []
        self.molecule_set = None
        self._set_tasks(tasks)

    def _set_tasks(self, tasks):
        """
        Parameters
        ----------
        tasks: dict
            The tasks field of the config yaml containing various tasks
            and their parameters.
        
        """
        for task, task_configs in tasks.items():
            try: 
                if task == 'compare_target_molecule':
                    loaded_task = CompareTargetMolecule(task_configs)    
                elif task == 'visualize_dataset':
                     loaded_task = VisualizeDataset(task_configs)
                elif task == 'show_property_variation_w_similarity':
                    loaded_task = ShowPropertyVariationWithSimilarity(
                                                                    task_configs)
                else:
                    print(f'{task} not recognized')
                    continue
            except IOError as e:
                print(f'Error in the config file for task: ', task)
                print('\n', e)
                exit(1)
            self.to_do.append(loaded_task)

        if len(self.to_do):
            print('No tasks were read. Exiting')
            exit(1)
    
    def _initialize_molecule_set(self, molecule_set_configs):
        """Initialize molecule_set attribute to a MoleculeSet object 
        based on parameters in the config file

        Parameters
        ----------
        molecule_set_configs: dict
            Configurations for initializing the MoleculeSet object.

        """
        molecule_database_src = molecule_set_configs.get('molecule_database', None)
        database_src_type = molecule_set_configs.get('molecule_database_source_type',
                                                None)
        if molecule_database_src is None:
            print('molecule_database field not set in config file')
            exit(1)
        is_verbose = molecule_set_configs.get('is_verbose', False)
        similarity_measure = molecule_set_configs.get('similarity_measure',
                                                'tanimoto_similarity')
        molecular_descriptor = molecule_set_configs.get('molecular_descriptor',
                                                    'morgan_fingerprint')
        self.molecule_set = MoleculeSet(
                                    molecule_database_src=molecule_database_src,
                                    molecule_database_src_type=database_src_type,
                                    similarity_measure=similarity_measure,
                                    molecular_descriptor=molecular_descriptor,
                                    is_verbose=is_verbose)

    def __call__(self, molecule_set_configs):
        """Launch all tasks from the queue.
                
        Parameters
        ----------
        molecule_set: Molecules object
            Molecules object of the molecule database.
        
        """
        self._initialize_molecule_set(molecule_set_configs)
        if self.molecule_set.is_verbose:
            print('Beginning tasks...')
        for task_id, task in enumerate(self.to_do):
            print(f'Task ({task_id + 1} / len(self.to_do)) {task}')
            task(self.molecule_set) 
       










        molecule_database = get_molecule_database(molecule_database_configs)
        for task, task_configs in tasks.items():
            if task == 'compare_target_molecule':
                target_molecule_smiles = task_configs.get('target_molecule_smiles')
                target_molecule_src = task_configs.get('target_molecule_src')
                if target_molecule_smiles:
                    target_molecule = Molecule(mol_smiles=target_molecule_smiles)
                elif target_molecule_src:
                    target_molecule = Molecule(mol_src=target_molecule_src)
                else:
                    raise IOError('Target molecule source is not specified '
                                f'for task {task}')
                save_to_file = task_configs.get('save_to_file', None)
                pdf_plot_kwargs = task_configs.get('plot_settings')
                compare_target_molecule(target_molecule=target_molecule,
                                        molecule_set=molecule_database,
                                        out_fpath=save_to_file,
                                        **pdf_plot_kwargs)
            elif task == 'visualize_dataset':
                visualize_dataset(molecule_database, task_configs)
            elif task == 'show_property_variation_w_similarity':
                show_property_variation_w_similarity(molecule_database,
                                                    task_configs)
            else:
                raise NotImplementedError(
                    f'{task} entered in the <<task>> field is not implemented')

        input("Press enter to terminate (plots will be closed).")

 
class Task(ABC):
    def __init__(self, configs):
        """
        Parameters
        ----------
        configs: dict
            parameters of the task
        
        """
        self.configs = deepcopy(configs)
    
    @abstractmethod
    def _extract_configs(self):
        pass

    @abstractmethod
    def __call__(self, molecule_set):
        pass

    @abstractmethod
    def __str__(self):
        pass


class CompareTargetMolecule(Task):
    def __init__(self, configs):
        super().__init__(configs)
        self.target_molecule = None
        self.log_fpath = None
        self.plot_settings = None
        self._verify_and_extract_configs()
            
    def _extract_configs(self):
        target_molecule_smiles = self.configs.get('target_molecule_smiles')
        target_molecule_src = self.configs.get('target_molecule_src')
        if target_molecule_smiles:
            self.target_molecule = Molecule(mol_smiles=target_molecule_smiles)
        elif target_molecule_src:
            self.target_molecule = Molecule(mol_src=target_molecule_src)
        else:
            raise IOError('Target molecule source is not specified')
        
        self.log_fpath = self.configs.get('save_to_file', None)
        if self.log_fpath is not None:
            log_dir = basename(self.log_fpath)
            makedirs(log_dir, exist_ok=True)
        
        self.plot_settings = self.configs.get('similarity_plot_settings', None)
    
    def __call__(self, molecule_set):
        """
        Compare a target molecule with molecular database in terms
        of similarity.
        Parameters
        ----------
        target_molecule: Molecule object
            Target molecule.
        molecule_set: MoleculeSet object
            Database of molecules to compare against.
        out_fpath: str
            Filepath to output results. If None, results are not saved and
            simply displayed to IO.


        """
        target_similarity = self.target_molecule.compare_to_molecule_set(
                                                                   molecule_set)
        ### shift to MoleculeSet
        most_similar_mol = molecule_set.molecule_database[
                                                np.argmax(target_similarity)]
        least_similar_mol = molecule_set.molecule_database[
                                                np.argmin(target_similarity)]
        ###############

        text_prompt = '***** '
        text_prompt += f'FOR MOLECULE {self.target_molecule.mol_text} *****'
        text_prompt += '\n\n'
        text_prompt += '****Maximum Similarity Molecule ****\n'
        text_prompt += f'Molecule: {most_similar_mol.mol_text}\n'
        text_prompt += 'Similarity: '
        text_prompt += str(max(target_similarity))
        text_prompt += '\n'
        text_prompt += '****Minimum Similarity Molecule ****\n'
        text_prompt += f'Molecule: {least_similar_mol.mol_text}\n'
        text_prompt += 'Similarity: '
        text_prompt += str(min(target_similarity))
        if self.log_fpath is None:
            print(text_prompt)
        else:
            if molecule_set.is_verbose:
                print(text_prompt)
            print('Writing to file ', self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(text_prompt)
        plot_density(target_similarity, **self.plot_settings)

    def __str__(self):
        return 'Task: Compare to a target molecule'


class VisualizeDataset(Task):
    def __init__(self, configs):
        super().__init__(configs)
        self.plot_settings = {}
        self._verify_and_extract_configs()
    
    def _extract_configs(self):
        self.plot_settings['heatmap_plot'] = self.configs.get(
                                            'heatmap_plot_settings', 
                                            None)
        self.plot_settings['pairwise_plot'] = self.configs.get(
                                            'pairwise_similarity_plot_settings',
                                            None)
        
    def __call__(self, molecule_set):
        """ Visualize essential properties of the dataset.

        Parameters
        ----------
        molecule_set: MoleculeSet object
            Molecular database initialized with the parameters.

        Plots Generated
        ---------------
        1. Heatmap of Molecular Similarity.
        2. PDF of the similarity distribution of the molecules in the database.

        """
        similarity_matrix = molecule_set.get_similarity_matrix()
        if molecule_set.is_verbose:
            print('Plotting similarity heatmap')
        plot_heatmap(similarity_matrix, self.plot_settings['heatmap_plot'])
        if molecule_set.is_verbose:
            print('Generating pairwise similarities')
        pairwise_similarity_vector = np.array([similarity_matrix[row, col]
                                            for row, col
                                            in zip(
                                                range(
                                                similarity_matrix.shape[0]),
                                                range(
                                                similarity_matrix.shape[1]))
                                            if row < col])
        if molecule_set.is_verbose:
            print('Plotting density of pairwise similarities')
        plot_density(pairwise_similarity_vector, 
                     self.plot_settings['pairwise_plot'])
    
    def __str__(self):
        return 'Task: Visualize a dataset'


class ShowPropertyVariationWithSimilarity(Task):
    def __init__(self, configs):
        super().__init__(configs)
        self.plot_settings = None
        self.log_fpath = None
        self._verify_and_extract_configs()

    def _extract_configs(self):
        self.plot_settings = {'xlabel': 'Reference Molecule Property',
                              'ylabel': 'Most Similar Molecule Property'
                              }
        self.plot_settings.update(self.configs.get('property_plot_settings', 
                                                   {}))
        
        self.log_fpath = self.configs.get('save_to_file', None)
        if self.log_fpath is not None:
            log_dir = basename(self.log_fpath)
            makedirs(log_dir, exist_ok=True)
    
    def __call__(self, molecule_set):
        """Plot the variation of molecular property with molecular fingerprint.

        Parameters
        ----------
        molecule_set: Molecules object
            Molecules object of the molecule database.

        """
        similar_mol_pairs = molecule_set.get_most_similar_pairs()
        dissimilar_mol_pairs = molecule_set.get_most_dissimilar_pairs()
        
        reference_mol_properties, similar_mol_properties = [], []
        for mol_pair in similar_mol_pairs:
            mol1_property = mol_pair[0].get_mol_property_val()
            mol2_property = mol_pair[1].get_mol_property_val()
            if mol1_property and mol2_property:
                reference_mol_properties.append(mol1_property)
                similar_mol_properties.append(mol2_property)
        reference_mol_properties, dissimilar_mol_properties = [], []
        for mol_pair in dissimilar_mol_pairs:
            mol1_property = mol_pair[0].get_mol_property_val()
            mol2_property = mol_pair[1].get_mol_property_val()
            if mol1_property and mol2_property:
                reference_mol_properties.append(mol1_property)
                dissimilar_mol_properties.append(mol2_property)
      
        if molecule_set.is_verbose:
            print('Plotting Responses of Similar Molecules')
        plot_parity(reference_mol_properties,
                    similar_mol_properties,
                    self.plot_settings)
        if molecule_set.is_verbose:
            print('Plotting Responses of Dissimilar Molecules')
        plot_parity(reference_mol_properties,
                    dissimilar_mol_properties,
                    self.plot_settings)

        #### Put in Molecule #####
        pearson_coff_of_responses = pearsonr(reference_mol_properties,
                                             similar_mol_properties)
        pearson_coff_of_dissimilar_responses = pearsonr(
                                                       reference_mol_properties,
                                                      dissimilar_mol_properties)
        ##############################
        text_prompt = 'Pearson Correlation in the properties of the ' \
                      'most similar molecules\n'
        text_prompt += '-' *  60
        text_prompt += '\n\n'
        text_prompt += f'{pearson_coff_of_responses[0]}'
        text_prompt += '\n'
        text_prompt += f'2 tailed p-value: {pearson_coff_of_responses[1]}'
        text_prompt += '\n\n\n\n'
        text_prompt = 'Pearson Correlation in the properties of the ' \
                      'most dissimilar molecules\n'
        text_prompt += '-' *  60
        text_prompt += '\n\n'
        text_prompt += f'{pearson_coff_of_dissimilar_responses[0]}'
        text_prompt += '\n'
        text_prompt += '2 tailed p-value: ' \
                       f'{pearson_coff_of_dissimilar_responses[1]}'
        if self.log_fpath is None:
            print(text_prompt)
        else:
            if molecule_set.is_verbose:
                print(text_prompt)
            print('Writing to file ', self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(text_prompt)
        
        def __str__(self):
            return 'Task: show variation of molecule property with similarity'


        



            




