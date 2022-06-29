from os import makedirs
from os.path import dirname
import matplotlib.pyplot as plt
from aimsim.chemical_datastructures import Molecule
import numpy as np

from aimsim.utils.plotting_scripts import plot_density
from aimsim.exceptions import InvalidConfigurationError
from .task import Task


class CompareTargetMolecule(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            if kwargs == {}:
                raise IOError(f"No config supplied for {str(self)}")
            else:
                configs = {}
                configs.update(kwargs)
        super().__init__(configs)
        self.target_molecule = None
        self.log_fpath = None
        self.plot_settings = None
        self._extract_configs()

    def _extract_configs(self):
        target_molecule_smiles = self.configs.get("target_molecule_smiles")
        target_molecule_src = self.configs.get("target_molecule_src")
        if target_molecule_smiles:
            self.target_molecule = Molecule(mol_smiles=target_molecule_smiles)
        elif target_molecule_src:
            self.target_molecule = Molecule(mol_src=target_molecule_src)
        else:
            raise IOError("Target molecule source is not specified")

        self.log_fpath = self.configs.get("log_file_path", None)
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

        self.plot_settings = self.configs.get("similarity_plot_settings", {})
        self.n_hits = self.configs.get("n_hits", 1)
        self.draw_molecules = self.configs.get("draw_molecules", False)

    def __call__(self, molecule_set):
        """
        Compare a target molecule with molecular database in terms
        of similarity.
        Args:
            molecule_set (AIMSim.chemical_datastructures Molecule): Target
                molecule.

        """
        most_similar_mols, sims = self.get_hits_similar_to(
            molecule_set=molecule_set)
        most_dissimilar_mols, dissims = self.get_hits_dissimilar_to(
            molecule_set=molecule_set)
        most_similar_mols = [molecule_set.molecule_database[mol_id]
                             for mol_id in most_similar_mols]
        most_dissimilar_mols = [molecule_set.molecule_database[mol_id]
                                for mol_id in most_dissimilar_mols]
        text_prompt = "***** "
        text_prompt += f"FOR MOLECULE {self.target_molecule.mol_text} *****"
        text_prompt += "\n\n"
        text_prompt += "****Maximum Similarity Molecules ****\n"
        for molecule, similarity in zip(most_similar_mols, sims):
            text_prompt += f"Molecule: {molecule.mol_text}\n"
            text_prompt += "Similarity: "
            text_prompt += str(similarity)
            if self.draw_molecules:
                molecule.draw(title=molecule.mol_text)

        text_prompt += "\n\n"
        text_prompt += "****Minimum Similarity Molecules ****\n"
        for molecule, similarity in zip(most_dissimilar_mols, dissims):
            text_prompt += f"Molecule: {molecule.mol_text}\n"
            text_prompt += "Similarity: "
            text_prompt += str(similarity)
            if self.draw_molecules:
                molecule.draw(title=molecule.mol_text)
        text_prompt += "\n\n\n"
        if self.log_fpath is None:
            print(text_prompt)
        else:
            if molecule_set.is_verbose:
                print(text_prompt)
            print("Writing to file ", self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(text_prompt)
        plot_density(self.similarities_, **self.plot_settings)

    def __str__(self):
        return "Task: Compare to a target molecule"

    def get_hits_similar_to(self, molecule_set=None):
        """Get sorted list of num_hits Molecule in the Set most
        similar to a query Molecule.This is defined as the sorted set
        (decreasing similarity)  of molecules with the highest
        (query_molecule, set_molecule) similarity.

        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                MoleculeSet object used to calculate sorted similarities.
                Only used if self.similarities or
                self.sorted_similarities not set.

        Returns:
            np.ndarray(int): Ids of most similar
                molecules in decreasing order of similarity.
            np.ndarray(float): Corresponding similarity values.

        """
        if not hasattr(self, 'sorted_similarities_'):
            if not hasattr(self, 'similarities_'):
                if molecule_set is None:
                    raise InvalidConfigurationError('MoleculeSet object not '
                                                    'passed for task')
                else:
                    self.similarities_ = molecule_set.compare_against_molecule(
                        self.target_molecule)
            self.sorted_similarities_ = np.argsort(self.similarities_)
        ids = np.array([self.sorted_similarities_[-1 - hit_id]
                        for hit_id in range(self.n_hits)])

        return ids, self.similarities_[ids]

    def get_hits_dissimilar_to(self, molecule_set=None):
        """Get sorted list of num_hits Molecule in the Set most
        dissimilar to a query Molecule.This is defined as the sorted set
        (decreasing dissimilarity)  of molecules with the highest
        (query_molecule, set_molecule) dissimilarity.

        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                MoleculeSet object used to calculate sorted similarities.
                Only used if self.similarities or
                self.sorted_similarities not set.

        Returns:
            np.ndarray(int): Ids of most similar molecules in decreasing
                order of dissimilarity.
            np.ndarray(float): Corresponding similarity values.

        """
        if not hasattr(self, 'sorted_similarities_'):
            if not hasattr(self, 'similarities_'):
                if molecule_set is None:
                    raise InvalidConfigurationError('MoleculeSet object not '
                                                    'passed for task')
                else:
                    self.similarities_ = molecule_set.compare_against_molecule(
                        self.target_molecule)
            self.sorted_similarities_ = np.argsort(self.similarities_)
        ids = np.array([self.sorted_similarities_[hit_id]
                        for hit_id in range(self.n_hits)])
        return ids, self.similarities_[ids]
