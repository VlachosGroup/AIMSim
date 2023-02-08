"""Class to call al tasks in sequence."""
import plotly.graph_objects as go
from aimsim.chemical_datastructures import MoleculeSet
from aimsim.exceptions import InvalidConfigurationError
from aimsim.tasks import *

import matplotlib.pyplot as plt


class TaskManager:
    def __init__(self, tasks):
        """Sequentially launches all the tasks from the configuration file.

        Args:
            tasks (dict): The tasks field of the config yaml containing various
                tasks and their parameters.
        """
        self.to_do = []
        self.molecule_set = None
        self._set_tasks(tasks)

    def _set_tasks(self, tasks):
        """
        Args:
            tasks (dict): The tasks field of the config yaml containing various
            tasks and their parameters.
        """
        for task, task_configs in tasks.items():
            try:
                if task == "compare_target_molecule":
                    loaded_task = CompareTargetMolecule(task_configs)
                elif task == "visualize_dataset":
                    loaded_task = VisualizeDataset(task_configs)
                elif task == "see_property_variation_w_similarity":
                    loaded_task = SeePropertyVariationWithSimilarity(
                        task_configs)
                elif task == "identify_outliers":
                    loaded_task = IdentifyOutliers(task_configs)
                elif task == "cluster":
                    loaded_task = ClusterData(task_configs)
                else:
                    print(f"{task} not recognized")
                    continue
                self.to_do.append(loaded_task)
            except InvalidConfigurationError as e:
                print(f"Error in the config file for task: ", task)
                print("\n", e)
                raise e

        if len(self.to_do) == 0:
            raise InvalidConfigurationError("No tasks were read, exiting.")

    def _initialize_molecule_set(self, molecule_set_configs):
        """Initialize molecule_set attribute to a MoleculeSet object
        based on parameters in the config file.

        Args:
            molecule_set_configs (dict): Configurations for initializing
                the MoleculeSet object.
        """
        molecule_database_src = molecule_set_configs.get(
            "molecule_database",
            None,
        )
        database_src_type = molecule_set_configs.get(
            "molecule_database_source_type", None
        )
        if molecule_database_src is None or database_src_type is None:
            print("molecule_database fields not set in config file")
            print(f"molecule_database: {molecule_database_src}")
            print(f"molecule_database_source_type: {database_src_type}")
            raise InvalidConfigurationError
        is_verbose = molecule_set_configs.get("is_verbose", False)
        n_threads = molecule_set_configs.get("n_workers", 1)
        similarity_measure = molecule_set_configs.get("similarity_measure",
                                                      'determine')
        fingerprint_type = molecule_set_configs.get('fingerprint_type',
                                                    'determine')
        fingerprint_params = molecule_set_configs.get('fingerprint_params', {})
        if similarity_measure == 'determine' or fingerprint_type == 'determine':
            subsample_subset_size = molecule_set_configs.get(
                'measure_id_subsample',
                0.05)
            only_valid_dist = molecule_set_configs.get(
                    'only_valid_dist',
                    True)
            if is_verbose:
                print('Determining best fingerprint_type / similarity_measure')
            measure_search = MeasureSearch(correlation_type='pearson')
            if similarity_measure == 'determine':
                similarity_measure = None
            if fingerprint_type == 'determine':
                fingerprint_type = None
                fingerprint_params = {}
            measure_search_molset_configs = {
                'molecule_database_src': molecule_database_src,
                'molecule_database_src_type': database_src_type,
                'is_verbose': is_verbose,
                'n_threads': n_threads,
            }

            best_measure = measure_search(
                molecule_set_configs=measure_search_molset_configs,
                similarity_measure=similarity_measure,
                fingerprint_type=fingerprint_type,
                fingerprint_params=fingerprint_params,
                subsample_subset_size=subsample_subset_size,
                show_top=5,
                only_metric=only_valid_dist)
            similarity_measure = best_measure.similarity_measure
            fingerprint_type = best_measure.fingerprint_type
            print(f'Chosen measure: {fingerprint_type} '
                  f'and {similarity_measure}.')

        sampling_ratio = molecule_set_configs.get("sampling_ratio", 1.)
        print(f'Choosing sampling ratio of {sampling_ratio} for tasks')
        self.molecule_set = MoleculeSet(
            molecule_database_src=molecule_database_src,
            molecule_database_src_type=database_src_type,
            similarity_measure=similarity_measure,
            fingerprint_type=fingerprint_type,
            fingerprint_params=fingerprint_params,
            is_verbose=is_verbose,
            n_threads=n_threads,
            sampling_ratio=sampling_ratio,
        )

    def __call__(self, molecule_set_configs):
        """Launch all tasks from the queue.

        Args:
            molecule_set_configs (dict): Configurations for the molecule_set.
        """
        self._initialize_molecule_set(molecule_set_configs)
        if self.molecule_set.is_verbose:
            print("Beginning tasks...")
        for task_id, task in enumerate(self.to_do):
            print(f"Task ({task_id + 1} / {len(self.to_do)}) {task}")
            try:
                task(self.molecule_set)
            except InvalidConfigurationError as e:
                print(
                    f"{task} could not be performed due to the "
                    f"following error: {e.message}"
                )
                continue
        plt.show()
