"""Subclass of Task that implements an IsolationForest to identify outliers."""
from .task import Task
from sklearn.ensemble import IsolationForest
import warnings


class IdentifyOutliers(Task):
    """Subclass of Task to identify outliers via an IsolationForest.

    Args:
        Task (abstract class): Parent abstract class.
    """

    def __init__(self, configs):
        super().__init__(configs)
        print(configs)
        self._extract_configs()

    def _extract_configs(self):
        self.output = self.configs.get('output', 'terminal')

    def __call__(self, molecule_set):
        """Iterates through all molecules in molecule_set, trains an IsolationForest, and identifies outliers.

        Args:
            molecule_set (list): List of RDKit molecule objects.
        """
        descs = []
        for molecule in molecule_set.molecule_database:
            descs.append(molecule.descriptor.to_numpy())
        iof = IsolationForest()
        iof.fit(descs)
        print(" ~"*10 + " Outlier Detection " + "~ "*10)
        for nmol, anomaly in zip(range(len(molecule_set.molecule_database)), iof.predict(descs)):
            if anomaly == -1:
                msg = "Molecule {} (name: {}) is a potential outlier ({:.2f} outlier score)".format(
                    nmol + 1, molecule.mol_text, iof.decision_function(descs[nmol].reshape(1, -1))[0])
                if self.output == 'terminal':
                    warnings.warn(msg)
                else:
                    with open(self.output + ".log", "a") as file:
                        file.write(msg + "\n")
        input("Outlier detection complete (enter to continue).")

    def __str__(self):
        return 'Task: Identify outliers.'
