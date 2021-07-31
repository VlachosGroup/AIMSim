"""Data clustering task."""
from os import makedirs
from os.path import dirname
import matplotlib.pyplot as plt
import yaml

from .task import Task
from molSim.exceptions import InvalidConfigurationError
from molSim.utils.plotting_scripts import plot_barchart, plot_scatter


class ClusterData(Task):
    def __init__(self, configs):
        if configs is None:
            raise IOError(f"No config supplied for {str(self)}")
        super().__init__(configs)
        self.n_clusters = None
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()

    def _extract_configs(self):
        self.n_clusters = self.configs["n_clusters"]
        self.clustering_method = self.configs.get("clustering_method", None)
        self.plot_settings = {
            "xlabel": "PC1",
            "ylabel": "PC2",
            "embedding": {"method": "pca"},
            "cluster_colors": [
                plt.cm.get_cmap("tab20", self.n_clusters)(cluster_id)
                for cluster_id in range(self.n_clusters)
            ],
        }
        self.plot_settings.update(self.configs.get("cluster_plot_settings", {}))

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
            molecule_set.cluster(n_clusters=self.n_clusters, 
                                 clustering_method=self.clustering_method)
        except InvalidConfigurationError as e:
            raise e
        mol_names = molecule_set.get_mol_names()
        cluster_labels = molecule_set.get_cluster_labels()
        cluster_grouped_mol_names = {}
        for cluster_id in range(self.n_clusters):
            cluster_grouped_mol_names[cluster_id] = mol_names[
                cluster_labels == cluster_id
            ].tolist()
        if molecule_set.is_verbose:
            print("Writing to file ", self.cluster_fpath)
        with open(self.cluster_fpath, "w") as fp:
            yaml.dump(cluster_grouped_mol_names, fp)

        plot_barchart(
            [_ for _ in range(self.n_clusters)],
            heights=[
                len(cluster_grouped_mol_names[cluster_id])
                for cluster_id in range(self.n_clusters)
            ],
            colors=self.plot_settings["cluster_colors"],
            xtick_labels=[_ for _ in range(self.n_clusters)],
            xlabel="Cluster Index",
            ylabel="Cluster Population",
        )

        if self.plot_settings["embedding"]["method"].lower() == "pca":
            reduced_features = molecule_set.get_transformed_descriptors(method_="pca")
        else:
            raise ValueError(
                "Embedding method "
                f'{self.plot_settings["embedding"]["method"]} '
                "not implemented."
            )
        plot_scatter(
            reduced_features[0],
            reduced_features[1],
            xlabel=self.plot_settings["xlabel"],
            ylabel=self.plot_settings["ylabel"],
            title=f"2-D projected space",
            plot_color=[
                self.plot_settings["cluster_colors"][cluster_num]
                for cluster_num in cluster_labels
            ],
        )

    def __str__(self):
        return "Task: Cluster data"
    

