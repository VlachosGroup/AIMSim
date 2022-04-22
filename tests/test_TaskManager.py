"""Test the TaskManager class."""
from aimsim.exceptions import InvalidConfigurationError
from aimsim.tasks.task_manager import TaskManager

import unittest
from unittest.mock import patch


class TestTaskManager(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_smiles = [
            "CCCCCCC",
            "CCCC",
            "CCC",
            "CO",
            "CN",
            "C1=CC=CC=C1",
            "CC1=CC=CC=C1",
            "C(=O)(N)N",
        ]

        property_seq = [i for i in range(len(self.test_smiles))]
        self.text_fpath = "temp_smiles_seq.txt"
        print(f"Creating text file {self.text_fpath}")
        with open(self.text_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)

        self.tasks_dict = {}
        self.inner_dict = {}
        self.inner_dict["plot_settings"] = {
            "plot_color": "green",
            "plot_title": "Entire Dataset",
        }
        self.inner_dict["pairwise_heatmap_settings"] = {
            "annotate": False,
            "cmap": "viridis",
        }
        self.tasks_dict["visualize_dataset"] = self.inner_dict
        self.tasks_dict["identify_outliers"] = {"output": "terminal"}
        self.tasks_dict["compare_target_molecule"] = {
            "target_molecule_smiles": "CCCC",
            "plot_settings": {
                "plot_color": "orange",
                "plot_title": "Compared to Target Molecule",
            },
            "identify_closest_furthest": {"out_file_path": "AIMSim-ui_output.txt"},
        }
        self.tasks_dict["see_property_variation_w_similarity"] = {
            "property_file": self.text_fpath,
            "most_dissimilar": True,
            "similarity_plot_settings": {"plot_color": "red"},
        }
        self.tasks_dict["cluster"] = {
            "n_clusters": 2,
        }
        self.tasks_dict["invalid_test_task"] = {
            "property_file": self.text_fpath,
            "most_dissimilar": True,
            "similarity_plot_settings": {"plot_color": "red"},
        }

    def test_no_tasks_task_manager(self):
        """Instant stop test for TaskManager class.
        """
        with self.assertRaises(InvalidConfigurationError):
            task_man = TaskManager(
                {}
            )

    def test_task_manager(self):
        """Complete run test for TaskManager class.
        """
        # this magical context manager sends all plots to a blank function
        with patch("aimsim.utils.plotting_scripts.plt.show") as test_plot:
            with patch("aimsim.utils.plotting_scripts.go.Figure.show") as test_int_plot:
                task_man = TaskManager(
                    tasks=self.tasks_dict
                )
                task_man(
                    molecule_set_configs={
                        'molecule_database': self.text_fpath,
                        'molecule_database_source_type': 'text',
                        'fingerprint_type': "morgan_fingerprint",
                        'similarity_measure': "tanimoto",
                        'is_verbose': False,
                    }
                )
                self.assertTrue(
                    test_plot.called or test_int_plot.called, "No plots were created.")


if __name__ == "__main__":
    unittest.main()
