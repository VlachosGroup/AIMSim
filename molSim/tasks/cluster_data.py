from os import makedirs
from os.path import basename

import matplotlib.pyplot as plt
import yaml

from .task import Task
from molSim.utils.plotting_scripts import plot_barchart


class ClusterData(Task):
    def __init__(self, configs):
        super().__init__(configs)
        self.n_clusters = None
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()
    
    def _extract_configs(self):
        self.n_clusters = self.configs['n_clusters']
        self.clustering_method = self.configs.get('clustering_method', None)
        self.plot_settings = {'xlabel': 'PC1',
                              'ylabel': 'PC2',
                              'embedding': {'method': 'pca'},
                              'cluster_colors': plt.cm.get_cmap(
                                                               'hsv', 
                                                               self.n_clusters)}
        self.plot_settings.update(self.configs.get('cluster_plot_settings', {}))
        
        self.log_fpath = self.configs.get('log_file_path', None)
        if self.log_fpath is not None:
            log_dir = basename(self.log_fpath)
            makedirs(log_dir, exist_ok=True)
        
        self.cluster_fpath = self.configs.get('cluster_file_path', None)
        if self.cluster_fpath is not None:
            cluster_dir = basename(self.cluster_fpath)
            makedirs(cluster_dir, exist_ok=True)
    
    def __call__(self, molecule_set):
        cluster_grouped_mol_names = molecule_set.cluster(
                                       n_clusters=self.n_clusters, 
                                       clustering_method=self.clustering_method)
        if molecule_set.is_verbose:
            print('Writing to file ', self.cluster_fpath)
        with open(self.cluster_fpath, "w") as fp:
            yaml.dump(cluster_grouped_mol_names, fp)
        
        plot_barchart([_ for _ in range(self.n_clusters)],
                      heights=[len(cluster_grouped_mol_names[cluster_id])
                               for cluster_id in range(self.n_clusters)],
                      colors=self.plot_settings['cluster_colors'],
                      xtick_labels=[_ for _ in range(self.n_clusters)],
                      xlabel='Cluster Index',
                      ylabel='Cluster Population')

    def __str__(self):
        return 'Task: Cluster data'
