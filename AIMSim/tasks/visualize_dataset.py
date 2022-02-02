"""Create similarity plots for the dataset."""
import matplotlib.pyplot as plt

from .task import Task
from AIMSim.utils.plotting_scripts import plot_density, plot_heatmap, \
    plot_scatter


class VisualizeDataset(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()  # all configs are optional
        configs.update(kwargs)
        super().__init__(configs)
        self.plot_settings = {}
        self._extract_configs()

    def _extract_configs(self):
        self.plot_settings = dict()
        self.plot_settings["heatmap_plot"] = self.configs.get(
            "heatmap_plot_settings", {}
        )
        self.plot_settings["pairwise_plot"] = self.configs.get(
            "similarity_plot_settings", {}
        )
        self.plot_settings["embeddings"] = {"method": "mds",
                                            "random_state": 42}
        self.plot_settings["embedding_plot"] = {
            "plot_color": 'red',
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

    def __call__(self, molecule_set):
        """Visualize essential properties of the dataset.

        Args:
            molecule_set(AIMSim.chemical_datastructures MoleculeSet):
                Molecular database.

        Plots Generated
        ---------------
        1. Heatmap of Molecular Similarity.
        2. PDF of the similarity distribution of the molecules in the database.

        """
        similarity_matrix = molecule_set.get_similarity_matrix()
        if molecule_set.is_verbose:
            print("Plotting similarity heatmap")
        plot_heatmap(similarity_matrix, **self.plot_settings["heatmap_plot"])
        if molecule_set.is_verbose:
            print("Generating pairwise similarities")
        pairwise_similarity_vector = molecule_set.get_pairwise_similarities()
        if molecule_set.is_verbose:
            print("Plotting density of pairwise similarities")
        plot_density(
            pairwise_similarity_vector,
            **self.plot_settings["pairwise_plot"],
        )
        plt.show()

        method_ = self.plot_settings["embedding_plot"]["embedding"]["method"]
        reduced_features = molecule_set.get_transformed_descriptors(
            method_=method_,
            n_components=2,
            **self.plot_settings["embedding_plot"]["embedding"]["params"])
        dimension_1 = reduced_features[:, 0]
        dimension_2 = reduced_features[:, 1]

        plot_scatter(
            dimension_1,
            dimension_2,
            xlabel=self.plot_settings["embedding_plot"]["xlabel"],
            ylabel=self.plot_settings["embedding_plot"]["ylabel"],
            title=self.plot_settings["embedding_plot"]["plot_title"],
            plot_color=self.plot_settings["embedding_plot"]["plot_color"],
            offset=0,
        )
        plt.show()

    def __str__(self):
        return "Task: Visualize a dataset"