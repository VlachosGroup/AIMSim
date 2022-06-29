"""Task to visualize property similarity over a dataset."""
from os import makedirs
from os.path import dirname

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from aimsim.utils.plotting_scripts import plot_parity
from aimsim.exceptions import InvalidConfigurationError
from .task import Task


class SeePropertyVariationWithSimilarity(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()  # all configs are optional
        configs.update(kwargs)
        super().__init__(configs)
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()

    def _extract_configs(self):
        """
        Raises:
        InvalidConfigurationError: If correlation_type does not match
                                   implemented types.
        """
        self.plot_settings = {"response": "response"}
        self.plot_settings.update(self.configs.get("property_plot_settings",
                                                   {}))
        self.log_fpath = self.configs.get("log_file_path", None)
        self.correlation_type = self.configs.get('correlation_type')
        if self.correlation_type is None:
            self.correlation_type = 'pearson'
        if self.correlation_type.lower() in ['pearson', 'linear']:
            self.correlation_fn = pearsonr
        else:
            raise InvalidConfigurationError(f'{self.correlation_type} correlation '
                                            f'not supported')
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

    def __call__(self, molecule_set):
        """Plot the variation of molecular property with molecular fingerprint.

        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.

        """
        ref_prop, similar_prop = self._get_ref_neighbor_properties(
            molecule_set)

        # scipy gives a remarkably unhelpful error if properties are not provided
        try:
            similar_correlation_ = self.correlation_fn(ref_prop, similar_prop)
        except ValueError as e:
            raise ValueError(
                "Unable to generate Property Simlilarty plot.\n" +
                "Check for properly formatted properties in your input file."
            )
        if molecule_set.is_verbose:
            print("Plotting Responses of Similar Molecules")
        plot_parity(
            ref_prop,
            similar_prop,
            xlabel=f'Reference molecule {self.plot_settings["response"]}',
            ylabel=f'Most similar molecule {self.plot_settings["response"]}',
            text="Correlation: {:.2f} (p-value {:.2f})".format(
                similar_correlation_[0], similar_correlation_[1]),
            **self.plot_settings,
        )

        ref_prop, dissimilar_prop = self._get_ref_neighbor_properties(
            molecule_set=molecule_set,
            nearest=False)
        dissimilar_correlation_ = self.correlation_fn(
            ref_prop, dissimilar_prop)
        if molecule_set.is_verbose:
            print("Plotting Responses of Dissimilar Molecules")
        plot_parity(
            ref_prop,
            dissimilar_prop,
            xlabel=f'Reference molecule {self.plot_settings["response"]}',
            ylabel=f'Most dissimilar molecule {self.plot_settings["response"]}',
            text="Correlation: {:.2f} (p-value {:.2f})".format(
                dissimilar_correlation_[0], dissimilar_correlation_[1]),
            **self.plot_settings,
        )

        text_prompt = (
            f"{self.correlation_type} correlation in the properties of the "
            "most similar molecules\n"
        )
        text_prompt += "-" * 60
        text_prompt += "\n\n"
        text_prompt += f"{similar_correlation_[0]}"
        text_prompt += "\n"
        text_prompt += f"2 tailed p-value: {similar_correlation_[1]}"
        text_prompt += "\n\n\n\n"
        text_prompt = (
            f"{self.correlation_type} in the properties of the "
            f"most dissimilar molecules\n"
        )
        text_prompt += "-" * 60
        text_prompt += "\n\n"
        text_prompt += f"{dissimilar_correlation_[0]}"
        text_prompt += "\n"
        text_prompt += "2 tailed p-value: "
        text_prompt += f"{dissimilar_correlation_[1]}"
        if self.log_fpath is None:
            print(text_prompt)
        else:
            if molecule_set.is_verbose:
                print(text_prompt)
            print("Writing to file ", self.log_fpath)
            with open(self.log_fpath, "w") as fp:
                fp.write(text_prompt)

    def get_property_correlations_in_most_similar(self, molecule_set):
        """Get the correlation between the property of molecules and their
        nearest (most similar) neighbors
        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.
        Return:
            (float): Correlation between properties.

        """
        ref_prop, similar_prop = self._get_ref_neighbor_properties(
            molecule_set)
        return self.correlation_fn(ref_prop, similar_prop)

    def get_property_correlations_in_most_dissimilar(self, molecule_set):
        """Get the correlation between the property of molecules and their
        furthest (most dissimilar) neighbors
        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.
        Return:
            (float): Correlation between properties.

        """
        ref_prop, dissimilar_prop = self._get_ref_neighbor_properties(
            molecule_set=molecule_set,
            nearest=False)
        return self.correlation_fn(ref_prop, dissimilar_prop)

    def _get_ref_neighbor_properties(self, molecule_set, nearest=True):
        """Get the properties of reference molecules and their nearest
        or furthest neighbors.
        Args:
            molecule_set (AIMSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.
            nearest (bool):
                If True nearest (most similar) neighbors are used,
                else furthest (most dissimilar). Default is True.
        Returns:
            (tuple): The first index is an array of reference mol
            properties and the second index is an array of the
            property of their respective neighbors.

        """
        if nearest:
            return molecule_set.get_property_of_most_similar()
        else:
            return molecule_set.get_property_of_most_dissimilar()

    def __str__(self):
        return "Task: see variation of molecule property with similarity"
