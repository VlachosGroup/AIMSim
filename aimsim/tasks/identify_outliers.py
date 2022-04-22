"""Subclass of Task that implements an IsolationForest to identify outliers."""
from .task import Task
from sklearn.ensemble import IsolationForest
import warnings
from aimsim.utils.plotting_scripts import plot_scatter_interactive
import matplotlib.pyplot as plt


class IdentifyOutliers(Task):
    """Subclass of Task to identify outliers via an IsolationForest.

    Args:
        Task (abstract class): Parent abstract class.
    """

    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()  # all configs are optional
        configs.update(kwargs)
        super().__init__(configs)
        self.plot_settings = {}
        self._extract_configs()

    def _extract_configs(self):
        self.plot_settings["pairwise_plot"] = self.configs.get(
            "pairwise_similarity_plot_settings", {}
        )
        self.output = self.configs.get("output", "terminal")
        self.plot_outlier = self.configs.get("plot_ouliers", True)
        self.random_state = self.configs.get("random_state", 42)

    def __call__(self, molecule_set):
        """Iterates through all molecules in molecule_set,
        trains an IsolationForest, and identifies outliers.

        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.

        """
        descs = []
        for molecule in molecule_set.molecule_database:
            descs.append(molecule.descriptor.to_numpy())
        iof = IsolationForest(random_state=self.random_state)
        iof.fit(descs)
        print("~" * 58)
        print(" ~" * 10 + " Outlier Detection " + "~ " * 10)
        print("~" * 58)
        outlier_idxs = []
        for nmol, anomaly in zip(
            range(len(molecule_set.molecule_database)), iof.predict(descs)
        ):
            if anomaly == -1:
                outlier_idxs.append(nmol)
                msg = (
                    "Molecule {} (name: {}) is a potential outlier "
                    "({:.2f} outlier score)".format(
                        nmol + 1,
                        molecule_set.molecule_database[nmol],
                        iof.decision_function(descs[nmol].reshape(1, -1))[0],
                    )
                )
                if self.output == "terminal":
                    warnings.warn(msg)
                else:
                    with open(self.output + ".log", "a") as file:
                        file.write(msg + "\n")
        if self.plot_outlier:
            reduced_features = molecule_set.get_transformed_descriptors(
                method_="pca")
            plot_scatter_interactive(
                reduced_features[:, 0],
                reduced_features[:, 1],
                outlier_idxs=outlier_idxs,
                title=f"2-D projected space",
                **self.plot_settings["pairwise_plot"],
            )

        print("~" * 58)
        print("Outlier detection complete.")
        print("~" * 58)

    def __str__(self):
        return "Task: Identify outliers"
