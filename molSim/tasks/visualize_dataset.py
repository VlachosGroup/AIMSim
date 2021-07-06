from os import makedirs
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import yaml


class VisualizeDataset(Task):
    def __init__(self, configs):
        super().__init__(configs)
        self.plot_settings = {}
        self._extract_configs()
    
    def _extract_configs(self):
        self.plot_settings['heatmap_plot'] = self.configs.get(
                                            'heatmap_plot_settings', {})
        self.plot_settings['pairwise_plot'] = self.configs.get(
                                            'pairwise_similarity_plot_settings',
                                            {})
        
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
        plot_heatmap(similarity_matrix, **self.plot_settings['heatmap_plot'])
        if molecule_set.is_verbose:
            print('Generating pairwise similarities')
        pairwise_similarity_vector = molecule_set.get_pairwise_similarities()
        if molecule_set.is_verbose:
            print('Plotting density of pairwise similarities')
        plot_density(pairwise_similarity_vector, 
                     **self.plot_settings['pairwise_plot'])
    
    def __str__(self):
        return 'Task: Visualize a dataset'