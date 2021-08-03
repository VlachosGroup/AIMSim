from os import makedirs
from os.path import dirname
import matplotlib.pyplot as plt
from molSim.chemical_datastructures import Molecule
import numpy as np
from molSim.utils.plotting_scripts import plot_density

from .task import Task


class CompareTargetMolecule(Task):
    def __init__(self, configs):
        if configs is None:
            raise IOError(f"No config supplied for {str(self)}")
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

    def __call__(self, molecule_set):
        """
        Compare a target molecule with molecular database in terms
        of similarity.
        Args:
            molecule_set (molSim.chemical_datastructures Molecule): Target
                molecule.
        """
        target_similarity = molecule_set.compare_against_molecule(self.target_molecule)
        most_similar_mol = molecule_set.molecule_database[np.argmax(target_similarity)]
        least_similar_mol = molecule_set.molecule_database[np.argmin(target_similarity)]
        text_prompt = "***** "
        text_prompt += f"FOR MOLECULE {self.target_molecule.mol_text} *****"
        text_prompt += "\n\n"
        text_prompt += "****Maximum Similarity Molecule ****\n"
        text_prompt += f"Molecule: {most_similar_mol.mol_text}\n"
        text_prompt += "Similarity: "
        text_prompt += str(max(target_similarity))
        text_prompt += "\n"
        text_prompt += "****Minimum Similarity Molecule ****\n"
        text_prompt += f"Molecule: {least_similar_mol.mol_text}\n"
        text_prompt += "Similarity: "
        text_prompt += str(min(target_similarity))
        if self.log_fpath is None:
            print(text_prompt)
        else:
            if molecule_set.is_verbose:
                print(text_prompt)
            print("Writing to file ", self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(text_prompt)
        plot_density(target_similarity, **self.plot_settings)
        plt.show()

    def __str__(self):
        return "Task: Compare to a target molecule"
