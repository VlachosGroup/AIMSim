"""Test the MeasureSearch class."""
from genericpath import exists
from os import remove, getcwd
from os.path import abspath, join, isdir, isfile
from shutil import rmtree
import unittest

import numpy as np

from aimsim.exceptions import InvalidConfigurationError
from aimsim.tasks.measure_search import MeasureSearch


class TestMeasureSearch(unittest.TestCase):

    test_smiles = [
        "CCCCCCC",
        "CCCC",
        "CCC",
        "CO",
        "CN",
        "C1=CC=CC=C1",
        "CC1=CC=CC=C1",
        "C(=O)(N)N",
    ]

    def smiles_seq_to_textfile(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a text file.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses.. Defaults to None.

        Returns:
            str: Path to created file.
        """
        text_fpath = "temp_smiles_seq.txt"
        print(f"Creating text file {text_fpath}")
        with open(text_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)
        return text_fpath

    def test_msearch_init(self):
        """Basic initialization test for the MeasureSearch class."""
        msearch = MeasureSearch()
        self.assertIsInstance(msearch, MeasureSearch)
        self.assertIsNone(msearch.log_fpath)
        self.assertIsInstance(msearch.__str__(), str)

    def test_msearch_init_error(self):
        """Erroneous confirgurations should raise a config error."""
        with self.assertRaises(InvalidConfigurationError):
            msearch = MeasureSearch(correlation_type="invalid name")

    def test_msearch_completion(self):
        """MeasureSearch should complete running, though output cannot be directly tested."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
            )
        except Exception as e:
            self.fail("MeasureSearch failed basic no-args test.")
        remove(text_fpath)

    def test_logfile_generation(self):
        """When configured to do so, MeasureSearch should create a log file."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        log_dir_name = "molSim_msearch_logs"
        log_dir_path = join(abspath(getcwd()), log_dir_name)
        log_file_path = join(log_dir_path, "logs.json")
        msearch = MeasureSearch(log_file_path=log_file_path)
        _ = msearch(
            molecule_set_configs={
                "molecule_database_src": text_fpath,
                "molecule_database_src_type": "text",
            },
            subsample_subset_size=1.0,
        )
        self.assertTrue(isfile(log_file_path))
        self.assertTrue(isdir(log_dir_path))
        rmtree(log_dir_path)

    def test_fixed_fprint(self):
        """MeasureSearch should search for ideal metric when fingerprint is already chosen."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                fingerprint_type="morgan_fingerprint",
            )
        except Exception as e:
            self.fail("MeasureSearch failed fixed fingerprint test.")
        remove(text_fpath)

    def test_only_metric_search(self):
        """Check that this configuration option is able to execute."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                only_metric=True,
            )
        except Exception as e:
            self.fail("MeasureSearch failed fixed metric test.")
        remove(text_fpath)

    def test_only_metric_fixed_measure_search(self):
        """Check that this configuration option is able to execute."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                only_metric=True,
                similarity_measure="tanimoto",
            )
        except Exception as e:
            self.fail("MeasureSearch failed fixed metric test.")
        remove(text_fpath)

    def test_only_metric_fixed_fprint_search(self):
        """Check that this configuration option is able to execute."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                only_metric=True,
                fingerprint_type="morgan_fingerprint",
            )
        except Exception as e:
            self.fail("MeasureSearch failed fixed metric test.")
        remove(text_fpath)

    def test_fixed_SimilarityMeasure(self):
        """MeasureSearch should search for ideal fingerprint when metric is
        already chosen.
        """
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                similarity_measure="tanimoto",
            )
        except Exception as e:
            self.fail("MeasureSearch failed fixed metric test.")
        remove(text_fpath)

    def test_verbose_output(self):
        """Ensure execution does not raise an exception with verbose and show_top."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                    "is_verbose": True,
                },
                subsample_subset_size=1.0,
                show_top=5,
            )
        except Exception as e:
            self.fail("MeasureSearch failed verbose output test.")
        remove(text_fpath)

        pass

    def test_max_optim_algo(self):
        """Measure search using the "maximum" optimization algorithm."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                optim_algo="max",
            )
        except Exception as e:
            self.fail("MeasureSearch failed max optim_algo test.")
        remove(text_fpath)

    def test_min_optim_algo(self):
        """Measure search using the "minimum" optimization algorithm."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        try:
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                optim_algo="min",
            )
        except Exception as e:
            self.fail("MeasureSearch failed min optim_algo test.")
        remove(text_fpath)

    def test_error_optim_algo(self):
        """Measure search should error with invalid optimization algorithm."""
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        msearch = MeasureSearch()
        with self.assertRaises(InvalidConfigurationError):
            msearch.get_best_measure(
                molecule_set_configs={
                    "molecule_database_src": text_fpath,
                    "molecule_database_src_type": "text",
                },
                subsample_subset_size=1.0,
                optim_algo="fake algo",
            )
        remove(text_fpath)


if __name__ == "__main__":
    unittest.main()
