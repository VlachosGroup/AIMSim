"""Create similarity plots for the dataset."""
from .task import Task
from molSim.utils.plotting_scripts import plot_density, plot_heatmap, plt


import pylustrator

pylustrator.start()


class VisualizeDataset(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()  # all configs are optional
        configs.update(kwargs)
        super().__init__(configs)
        self.plot_settings = {}
        self._extract_configs()

    def _extract_configs(self):
        self.plot_settings["heatmap_plot"] = self.configs.get(
            "heatmap_plot_settings", {}
        )
        self.plot_settings["pairwise_plot"] = self.configs.get(
            "similarity_plot_settings", {}
        )

    def __call__(self, molecule_set):
        """Visualize essential properties of the dataset.

        Args:
            molecule_set(molSim.chemical_datastructures MoleculeSet):
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

    def __str__(self):
        return "Task: Visualize a dataset"
