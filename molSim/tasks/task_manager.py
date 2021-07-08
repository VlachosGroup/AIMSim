from molSim.chemical_datastructures import MoleculeSet
from molSim.tasks import CompareTargetMolecule, VisualizeDataset
from molSim.tasks import ShowPropertyVariationWithSimilarity


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
                self.to_do.append(loaded_task)
            except IOError as e:
                print(f'Error in the config file for task: ', task)
                print('\n', e)
                exit(1)

        if len(self.to_do) == 0:
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
        molecule_database_src = molecule_set_configs.get('molecule_database',
                                                         None)
        database_src_type = molecule_set_configs.get(
                                                'molecule_database_source_type',
                                                None)
        if molecule_database_src is None or database_src_type is None:
            print('molecule_database fields not set in config file')
            print(f'molecule_database: {molecule_database_src}')
            print(f'molecule_database_source_type: {database_src_type}')
            exit(1)
        is_verbose = molecule_set_configs.get('is_verbose', False)
        n_threads = molecule_set_configs.get('n_threads', 1)
        similarity_measure = molecule_set_configs.get('similarity_measure',
                                                      'tanimoto_similarity')
        molecular_descriptor = molecule_set_configs.get('molecular_descriptor',
                                                        'morgan_fingerprint')
        self.molecule_set = MoleculeSet(
                                molecule_database_src=molecule_database_src,
                                molecule_database_src_type=database_src_type,
                                similarity_measure=similarity_measure,
                                molecular_descriptor=molecular_descriptor,
                                is_verbose=is_verbose,
                                n_threads=n_threads)

    def __call__(self, molecule_set_configs):
        """Launch all tasks from the queue.
                
        Parameters
        ----------
        molecule_set_configs: dict
        
        """
        self._initialize_molecule_set(molecule_set_configs)
        if self.molecule_set.is_verbose:
            print('Beginning tasks...')
        for task_id, task in enumerate(self.to_do):
            print(f'Task ({task_id + 1} / {len(self.to_do)}) {task}')
            task(self.molecule_set) 
        input("Press enter to terminate (plots will be closed).")



            




