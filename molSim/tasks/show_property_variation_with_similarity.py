"""Task to visualize property similarity over a dataset."""
from os import makedirs
from os.path import dirname
from scipy.stats import pearsonr
from molSim.utils.plotting_scripts import plot_parity

from .task import Task


class ShowPropertyVariationWithSimilarity(Task):
    def __init__(self, configs):
        if configs is None:
            configs = dict()  # all configs are optional
        super().__init__(configs)
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()

    def _extract_configs(self):
        self.plot_settings = {"response": "response"}
        self.plot_settings.update(self.configs.get("property_plot_settings", {}))

        self.log_fpath = self.configs.get("log_file_path", None)
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

    def __call__(self, molecule_set):
        """Plot the variation of molecular property with molecular fingerprint.

        Args:
            molecule_set (molSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.

        """
        similar_mol_pairs = molecule_set.get_most_similar_pairs()
        dissimilar_mol_pairs = molecule_set.get_most_dissimilar_pairs()

        reference_mol_properties, similar_mol_properties = [], []
        for mol_pair in similar_mol_pairs:
            mol1_property = mol_pair[0].get_mol_property_val()
            mol2_property = mol_pair[1].get_mol_property_val()
            if mol1_property and mol2_property:
                reference_mol_properties.append(mol1_property)
                similar_mol_properties.append(mol2_property)
        dissimilar_reference_mol_properties, dissimilar_mol_properties = [], []
        for mol_pair in dissimilar_mol_pairs:
            mol1_property = mol_pair[0].get_mol_property_val()
            mol2_property = mol_pair[1].get_mol_property_val()
            if mol1_property and mol2_property:
                dissimilar_reference_mol_properties.append(mol1_property)
                dissimilar_mol_properties.append(mol2_property)

        if molecule_set.is_verbose:
            print("Plotting Responses of Similar Molecules")

        plot_parity(
            reference_mol_properties,
            similar_mol_properties,
            xlabel=f'Reference molecule {self.plot_settings["response"]}',
            ylabel=f'Most similar molecule {self.plot_settings["response"]}',
            **self.plot_settings,
        )
        if molecule_set.is_verbose:
            print("Plotting Responses of Dissimilar Molecules")
        plot_parity(
            dissimilar_reference_mol_properties,
            dissimilar_mol_properties,
            xlabel=f'Reference molecule {self.plot_settings["response"]}',
            ylabel=f'Most dissimilar molecule {self.plot_settings["response"]}',
            **self.plot_settings,
        )

        pearson_coff_of_responses = pearsonr(
            reference_mol_properties, similar_mol_properties
        )
        pearson_coff_of_dissimilar_responses = pearsonr(
            dissimilar_reference_mol_properties, dissimilar_mol_properties
        )
        text_prompt = ("Pearson Correlation in the properties of the "
                       "most similar molecules\n")
        text_prompt += "-" * 60
        text_prompt += "\n\n"
        text_prompt += f"{pearson_coff_of_responses[0]}"
        text_prompt += "\n"
        text_prompt += f"2 tailed p-value: {pearson_coff_of_responses[1]}"
        text_prompt += "\n\n\n\n"
        text_prompt = (
            "Pearson Correlation in the properties of the "
            "most dissimilar molecules\n"
        )
        text_prompt += "-" * 60
        text_prompt += "\n\n"
        text_prompt += f"{pearson_coff_of_dissimilar_responses[0]}"
        text_prompt += "\n"
        text_prompt += "2 tailed p-value: " \
                       f"{pearson_coff_of_dissimilar_responses[1]}"
        if self.log_fpath is None:
            print(text_prompt)
        else:
            if molecule_set.is_verbose:
                print(text_prompt)
            print("Writing to file ", self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(text_prompt)

    def __str__(self):
        return "Task: show variation of molecule property with similarity"
