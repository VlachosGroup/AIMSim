"""Data clustering task."""
from os import makedirs
from os.path import dirname

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import yaml

from .task import Task
from aimsim.exceptions import InvalidConfigurationError
from aimsim.utils.plotting_scripts import plot_barchart, plot_density
from aimsim.utils.plotting_scripts import plot_scatter_interactive


class ClusterData(Task):
    def __init__(self, configs=None, **kwargs):
        """Constructor for the ClusterData class.

        Args:
            configs(dict): Dictionary of configurations. Default is None.
            **kwargs: Keyword arguments to modify configs fields.
        Notes:
            The configuration structure with default values are:
            'n_clusters' (int):
            'clustering_method' (str): None
            cluster_plot_settings: {'cluster_colors' (list):
                                                         colors from tab20 cmap,
                                    'response' (str): 'Response'}

            embedding_plot_settings:
                    {'plot_color': 'red'
                     'plot_title': '2-D projected space',
                     'xlabel': 'Dimension 1',
                     'ylabel': 'Dimension 2',
                     'embedding': {'method': "mds",
                                    'params': {'random_state': 42}}}

        """
        if configs is None:
            if kwargs == {}:
                raise IOError(f"No config supplied for {str(self)}")
            else:
                configs = {}
                configs.update(kwargs)
        super().__init__(configs)
        self.n_clusters = None
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()

    def _extract_configs(self):
        self.n_clusters = self.configs["n_clusters"]
        self.clustering_method = self.configs.get("clustering_method", None)
        self.plot_settings = dict()
        self.plot_settings["cluster_plot"] = {
            "cluster_colors": [
                rgb2hex(plt.cm.get_cmap("tab20", self.n_clusters)(cluster_id))
                for cluster_id in range(self.n_clusters)],
            "response": "Response",
        }

        self.plot_settings["cluster_plot"].update(
            self.configs.get("cluster_plot_settings", {}))
        self.plot_settings["embedding_plot"] = {
            "plot_title": f"2-D projected space",
            "xlabel": "Dimension 1",
            "ylabel": "Dimension 2",
            "embedding": {"method": "mds",
                          "params": {"random_state": 42, }
                          },
        }
        self.plot_settings["embedding_plot"].update(self.configs.get(
            "embedding_plot_settings",
            {}))

        self.log_fpath = self.configs.get("log_file_path", None)
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

        self.cluster_fpath = self.configs.get("cluster_file_path", None)
        if self.cluster_fpath is not None:
            cluster_dir = dirname(self.cluster_fpath)
            makedirs(cluster_dir, exist_ok=True)

    def __call__(self, molecule_set):
        try:
            molecule_set.cluster(
                n_clusters=self.n_clusters,
                clustering_method=self.clustering_method
            )
        except InvalidConfigurationError as e:
            raise e
        mol_names = molecule_set.get_mol_names()
        mol_properties = molecule_set.get_mol_properties()
        cluster_labels = molecule_set.get_cluster_labels()
        cluster_grouped_mol_names = {}
        cluster_grouped_mol_properties = {}
        for cluster_id in range(self.n_clusters):
            cluster_grouped_mol_names[cluster_id] = mol_names[
                cluster_labels == cluster_id
            ].tolist()
            if mol_properties is not None:
                cluster_grouped_mol_properties[cluster_id] = mol_properties[
                    cluster_labels == cluster_id
                ].tolist()

        if self.cluster_fpath is not None:
            print("Writing to file ", self.cluster_fpath)
            with open(self.cluster_fpath, "w") as fp:
                yaml.dump(cluster_grouped_mol_names, fp)
                if cluster_grouped_mol_properties != {}:
                    yaml.dump('Properties By Cluster', fp)
                    yaml.dump(cluster_grouped_mol_properties, fp)

        if self.log_fpath is not None:
            print("Writing to file ", self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(
                    f'Embedding method '
                    f'{self.plot_settings["embedding_plot"]["embedding"]["method"]}. '
                    f'random seed '
                    f'{self.plot_settings["embedding_plot"]["embedding"]["params"]["random_state"]}')

        plot_barchart(
            [_ for _ in range(self.n_clusters)],
            heights=[
                len(cluster_grouped_mol_names[cluster_id])
                for cluster_id in range(self.n_clusters)
            ],
            colors=self.plot_settings["cluster_plot"]["cluster_colors"],
            xtick_labels=[_ for _ in range(self.n_clusters)],
            xlabel="Cluster Index",
            ylabel="Cluster Population",
        )
        if mol_properties is not None:
            densities = []
            for cluster_id in range(self.n_clusters):
                densities.append(cluster_grouped_mol_properties[cluster_id])
            plot_density(
                densities=densities,
                n_densities=self.n_clusters,
                legends=['Cluster'+str(_)
                         for _ in range(self.n_clusters)],
                plot_color=self.plot_settings[
                    "cluster_plot"]["cluster_colors"],
                legend_fontsize=20,
                xlabel=self.plot_settings["cluster_plot"]["response"],
                ylabel='Density',
                shade=True,
            )
        method_ = self.plot_settings["embedding_plot"]["embedding"]["method"]
        reduced_features = molecule_set.get_transformed_descriptors(
            method_=method_,
            n_components=2,
            **self.plot_settings["embedding_plot"]["embedding"].get("params", {}))
        dimension_1 = reduced_features[:, 0]
        dimension_2 = reduced_features[:, 1]

        plot_scatter_interactive(
            dimension_1,
            dimension_2,
            cluster_memberships=cluster_labels,
            xlabel=self.plot_settings["embedding_plot"]["xlabel"],
            ylabel=self.plot_settings["embedding_plot"]["ylabel"],
            title=self.plot_settings["embedding_plot"]['plot_title'],
            hover_names=mol_names,
            cluster_colors=self.plot_settings["cluster_plot"]["cluster_colors"],
        )

    def __str__(self):
        return "Task: Cluster data"
