"""Task to visualize property similarity over a dataset."""
from os import makedirs
from os.path import dirname

import pylustrator
from scipy.stats import pearsonr

from molSim.utils.plotting_scripts import plot_parity, plt
from molSim.exceptions import InvalidConfigurationError
from .task import Task


pylustrator.start()


class ShowPropertyVariationWithSimilarity(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()  # all configs are optional
        configs.update(kwargs)
        super().__init__(configs)
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()

    def _extract_configs(self):
        self.plot_settings = {"response": "response"}
        self.plot_settings.update(self.configs.get("property_plot_settings",
                                                   {}))

        self.log_fpath = self.configs.get("log_file_path", None)
        self.correlation_type = self.configs.get("correlation_type",
                                                  "pearson").lower()
        if self.correlation_type in ['pearson', 'linear']:
            self.correlation_fn = pearsonr
        else:
            raise InvalidConfigurationError(f'{correlation_type} correlation '
                                            f'not supported')
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

    def __call__(self, molecule_set):
        """Plot the variation of molecular property with molecular fingerprint.

        Args:
            molecule_set (molSim.chemical_datastructures MoleculeSet):
                Molecules object of the molecule database.

        """
        ref_prop, similar_prop = molecule_set.get_property_of_most_similar()
        similar_correlation_ = self.correlation_fn(ref_prop, similar_prop)
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

        ref_prop, dissimilar_prop = molecule_set.get_property_of_most_dissimilar()
        dissimilar_correlation_ = self.correlation_fn(ref_prop, dissimilar_prop)
        if molecule_set.is_verbose:
            print("Plotting Responses of Dissimilar Molecules")
        plot_parity(
            dissimilar_reference_mol_properties,
            dissimilar_mol_properties,
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
        plt.show()

    def get_property_correlations_in_most_similar(self, molecule_set):
        ref_prop, similar_prop = molecule_set.get_property_of_most_similar()
        return self.correlation_fn(ref_prop, similar_prop)

    def get_property_correlations_in_most_dissimilar(self, molecule_set):
        ref_prop, dissimilar_prop = molecule_set.get_property_of_most_dissimilar()
        return self.correlation_fn(ref_prop, dissimilar_prop)

    def __str__(self):
        return "Task: show variation of molecule property with similarity"


