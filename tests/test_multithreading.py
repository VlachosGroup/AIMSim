""" Test multithreading to ensure consistent behavior with serial implementation."""
import unittest
import warnings
from os import remove
from os.path import exists, join
import numpy as np
from aimsim.chemical_datastructures import MoleculeSet
from time import time
from tabulate import tabulate


class TestMultithreading(unittest.TestCase):
    """Unit tests to ensure consistency when running AIMSim as a single process
    or when using multiprocessing.
    """

    @classmethod
    def setUpClass(self):
        """Create a SMILES database to use for comparisons and
        find the similarity matrices and execution times.
        """
        if not exists(".speedup-test"):
            print("Speedup and Efficiency tests DISABLED.")
            self.NO_SPEEDUP_TEST = True
        else:
            self.NO_SPEEDUP_TEST = False
            self.N_REPLICATES = 2
            warnings.warn(
                "Speedup and Efficiency tests ENABLED, expect long runtime.",
                ResourceWarning,
            )
        print(" ~ ~ Testing Multithreading ~ ~ ", flush=True)

        # basic consistency tests
        self.text_fpath = "temp_multithread_smiles_seq.txt"
        print(f"Creating text file {self.text_fpath}", flush=True)
        with open(self.text_fpath, "w") as file:
            for smiles in ["C", "CC", "CCC", "O", "CCCC", "CO", "CCOCC"]:
                file.write(smiles + "\n")
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=1,
            fingerprint_type="morgan_fingerprint",
        )
        self.correct_similarity_matrix = test_molecule_set.get_similarity_matrix()

        if self.NO_SPEEDUP_TEST:
            return
        with open(join("tests", "data", "combinatorial_1.txt"), "r") as file:
            data = file.readlines()
            _100_molecules = data[1:102]
            _500_molecules = data[1:502]
            _1000_molecules = data[1:1002]
            _5000_molecules = data[1:5002]
            _10000_molecules = data[1:10002]
            _15000_molecules = data[1:15002]

        # data used for speedup and efficiency tests
        self._100_molecules_fpath = "temp_multithread_speedup_100.txt"
        print(f"Creating text file {self._100_molecules_fpath}", flush=True)
        with open(self._100_molecules_fpath, "w") as file:
            for smiles in _100_molecules:
                file.write(smiles)
        print("Running 100 molecules with 1 process.", flush=True)
        self._100_molecules_serial_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=1,
                fingerprint_type="morgan_fingerprint",
            )
            # This creates a running average across all of the replicates.
            self._100_molecules_serial_time += (time() -
                                                start) / self.N_REPLICATES

        self._500_molecules_fpath = "temp_multithread_speedup_500.txt"
        print(f"Creating text file {self._500_molecules_fpath}", flush=True)
        with open(self._500_molecules_fpath, "w") as file:
            for smiles in _500_molecules:
                file.write(smiles)
        print("Running 500 molecules with 1 process.", flush=True)
        self._500_molecules_serial_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=1,
                fingerprint_type="morgan_fingerprint",
            )
            self._500_molecules_serial_time += (time() -
                                                start) / self.N_REPLICATES

        self._1000_molecules_fpath = "temp_multithread_speedup_1000.txt"
        print(f"Creating text file {self._1000_molecules_fpath}", flush=True)
        with open(self._1000_molecules_fpath, "w") as file:
            for smiles in _1000_molecules:
                file.write(smiles)
        print("Running 1000 molecules with 1 process.", flush=True)
        self._1000_molecules_serial_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=1,
                fingerprint_type="morgan_fingerprint",
            )
            self._1000_molecules_serial_time += (
                time() - start) / self.N_REPLICATES

        self._5000_molecules_fpath = "temp_multithread_speedup_5000.txt"
        print(f"Creating text file {self._5000_molecules_fpath}", flush=True)
        with open(self._5000_molecules_fpath, "w") as file:
            for smiles in _5000_molecules:
                file.write(smiles)
        print("Running 5000 molecules with 1 process.", flush=True)
        self._5000_molecules_serial_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=1,
                fingerprint_type="morgan_fingerprint",
            )
            self._5000_molecules_serial_time += (
                time() - start) / self.N_REPLICATES

        self._10000_molecules_fpath = "temp_multithread_speedup_10000.txt"
        print(f"Creating text file {self._10000_molecules_fpath}", flush=True)
        with open(self._10000_molecules_fpath, "w") as file:
            for smiles in _10000_molecules:
                file.write(smiles)
        print("Running 10000 molecules with 1 process.", flush=True)
        self._10000_molecules_serial_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=1,
                fingerprint_type="morgan_fingerprint",
            )
            self._10000_molecules_serial_time += (
                time() - start) / self.N_REPLICATES

        self._15000_molecules_fpath = "temp_multithread_speedup_15000.txt"
        print(f"Creating text file {self._15000_molecules_fpath}", flush=True)
        with open(self._15000_molecules_fpath, "w") as file:
            for smiles in _15000_molecules:
                file.write(smiles)
        print("Running 15000 molecules with 1 process.", flush=True)
        self._15000_molecules_serial_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=1,
                fingerprint_type="morgan_fingerprint",
            )
            self._15000_molecules_serial_time += (
                time() - start) / self.N_REPLICATES

        # data used for speedup and efficiency test 2
        print("Running 100 molecules with 1 process.", flush=True)
        self._100_molecules_serial_time_2 = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=1,
                fingerprint_type="topological_fingerprint",
            )
            self._100_molecules_serial_time_2 += (
                time() - start) / self.N_REPLICATES

        print("Running 500 molecules with 1 process.", flush=True)
        self._500_molecules_serial_time_2 = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=1,
                fingerprint_type="topological_fingerprint",
            )
            self._500_molecules_serial_time_2 += (
                time() - start) / self.N_REPLICATES

        print("Running 1000 molecules with 1 process.", flush=True)
        self._1000_molecules_serial_time_2 = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=1,
                fingerprint_type="topological_fingerprint",
            )
            self._1000_molecules_serial_time_2 += (
                time() - start) / self.N_REPLICATES

        print("Running 5000 molecules with 1 process.", flush=True)
        self._5000_molecules_serial_time_2 = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=1,
                fingerprint_type="topological_fingerprint",
            )
            self._5000_molecules_serial_time_2 += (
                time() - start) / self.N_REPLICATES

        print("Running 10000 molecules with 1 process.", flush=True)
        self._10000_molecules_serial_time_2 = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=1,
                fingerprint_type="topological_fingerprint",
            )
            self._10000_molecules_serial_time_2 += (
                time() - start) / self.N_REPLICATES

        print("Running 15000 molecules with 1 process.", flush=True)
        self._15000_molecules_serial_time_2 = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=1,
                fingerprint_type="topological_fingerprint",
            )
            self._15000_molecules_serial_time_2 += (
                time() - start) / self.N_REPLICATES

    def test_multithreading_autoconfig(self):
        """
        Ensure that MoleculeSet can automatically configure multiprocessing..
        """
        try:
            MoleculeSet(
                molecule_database_src=self.text_fpath,
                molecule_database_src_type="text",
                is_verbose=True,
                similarity_measure="tanimoto",
                n_threads='auto',
                fingerprint_type="morgan_fingerprint",
            )
        except Exception as e:
            self.fail("Multiprocessing automatic configuration failed.")

    def test_multithreading_consistency_2_threads(self):
        """
        Ensure that the similarity matrix produced with 2 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=2,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using two threads.",
        )

    def test_multithreading_consistency_3_threads(self):
        """
        Ensure that the similarity matrix produced with 3 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=3,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using three threads.",
        )

    def test_multithreading_consistency_4_threads(self):
        """
        Ensure that the similarity matrix produced with 4 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=4,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using four threads.",
        )

    def test_multithreading_consistency_5_threads(self):
        """
        Ensure that the similarity matrix produced with 5 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=5,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using five threads.",
        )

    def test_multithreading_consistency_6_threads(self):
        """
        Ensure that the similarity matrix produced with 6 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=6,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using six threads.",
        )

    def test_multithreading_consistency_7_threads(self):
        """
        Ensure that the similarity matrix produced with 7 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=7,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using seven threads (equal to the number of molecules).",
        )

    def test_multithreading_consistency_10_threads(self):
        """
        Ensure that the similarity matrix produced with 10 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure="tanimoto",
            n_threads=10,
            fingerprint_type="morgan_fingerprint",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_molecule_set.get_similarity_matrix(),
                self.correct_similarity_matrix,
            ),
            "Similarity matrix not equal when using ten threads (more than the number of molecules).",
        )

    def test_speedup_efficiency_tanimoto(self):
        """
        Evaluate the speedup and efficieny of the multiprocessing approach.
        """
        if self.NO_SPEEDUP_TEST:
            return
        print("~" * 10, "\n", "Speedup and Efficiency Test 1\n",
              "~" * 10, flush=True)
        # 100 molecules
        print("Running 100 molecules with 2 processes.", flush=True)
        _100_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=2,
                fingerprint_type="morgan_fingerprint",
            )
            _100_molecules_2_process_time += (time() -
                                              start) / self.N_REPLICATES
        _100_molecules_2_process_speedup = (
            self._100_molecules_serial_time / _100_molecules_2_process_time
        )
        _100_molecules_2_process_efficiency = _100_molecules_2_process_speedup / 2

        print("Running 100 molecules with 4 processes.", flush=True)
        _100_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=4,
                fingerprint_type="morgan_fingerprint",
            )
            _100_molecules_4_process_time += (time() -
                                              start) / self.N_REPLICATES
        _100_molecules_4_process_speedup = (
            self._100_molecules_serial_time / _100_molecules_4_process_time
        )
        _100_molecules_4_process_efficiency = _100_molecules_4_process_speedup / 4

        print("Running 100 molecules with 8 processes.", flush=True)
        _100_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=8,
                fingerprint_type="morgan_fingerprint",
            )
            _100_molecules_8_process_time += (time() -
                                              start) / self.N_REPLICATES
        _100_molecules_8_process_speedup = (
            self._100_molecules_serial_time / _100_molecules_8_process_time
        )
        _100_molecules_8_process_efficiency = _100_molecules_8_process_speedup / 8

        # 500 molecules
        print("Running 500 molecules with 2 processes.", flush=True)
        _500_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=2,
                fingerprint_type="morgan_fingerprint",
            )
            _500_molecules_2_process_time += (time() -
                                              start) / self.N_REPLICATES
        _500_molecules_2_process_speedup = (
            self._500_molecules_serial_time / _500_molecules_2_process_time
        )
        _500_molecules_2_process_efficiency = _500_molecules_2_process_speedup / 2

        print("Running 500 molecules with 4 processes.", flush=True)
        _500_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=4,
                fingerprint_type="morgan_fingerprint",
            )
            _500_molecules_4_process_time += (time() -
                                              start) / self.N_REPLICATES
        _500_molecules_4_process_speedup = (
            self._500_molecules_serial_time / _500_molecules_4_process_time
        )
        _500_molecules_4_process_efficiency = _500_molecules_4_process_speedup / 4

        print("Running 500 molecules with 8 processes.", flush=True)
        _500_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=8,
                fingerprint_type="morgan_fingerprint",
            )
            _500_molecules_8_process_time += (time() -
                                              start) / self.N_REPLICATES
        _500_molecules_8_process_speedup = (
            self._500_molecules_serial_time / _500_molecules_8_process_time
        )
        _500_molecules_8_process_efficiency = _500_molecules_8_process_speedup / 8

        # 1000 molecules
        print("Running 1000 molecules with 2 processes.", flush=True)
        _1000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=2,
                fingerprint_type="morgan_fingerprint",
            )
            _1000_molecules_2_process_time += (time() -
                                               start) / self.N_REPLICATES
        _1000_molecules_2_process_speedup = (
            self._1000_molecules_serial_time / _1000_molecules_2_process_time
        )
        _1000_molecules_2_process_efficiency = _1000_molecules_2_process_speedup / 2

        print("Running 1000 molecules with 4 processes.", flush=True)
        _1000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=4,
                fingerprint_type="morgan_fingerprint",
            )
            _1000_molecules_4_process_time += (time() -
                                               start) / self.N_REPLICATES
        _1000_molecules_4_process_speedup = (
            self._1000_molecules_serial_time / _1000_molecules_4_process_time
        )
        _1000_molecules_4_process_efficiency = _1000_molecules_4_process_speedup / 4

        print("Running 1000 molecules with 8 processes.", flush=True)
        _1000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=8,
                fingerprint_type="morgan_fingerprint",
            )
            _1000_molecules_8_process_time += (time() -
                                               start) / self.N_REPLICATES
        _1000_molecules_8_process_speedup = (
            self._1000_molecules_serial_time / _1000_molecules_8_process_time
        )
        _1000_molecules_8_process_efficiency = _1000_molecules_8_process_speedup / 8

        print("Running 5000 molecules with 2 processes.", flush=True)
        # 5000 molecules
        _5000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=2,
                fingerprint_type="morgan_fingerprint",
            )
            _5000_molecules_2_process_time += (time() -
                                               start) / self.N_REPLICATES
        _5000_molecules_2_process_speedup = (
            self._5000_molecules_serial_time / _5000_molecules_2_process_time
        )
        _5000_molecules_2_process_efficiency = _5000_molecules_2_process_speedup / 2

        print("Running 5000 molecules with 4 processes.", flush=True)
        _5000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=4,
                fingerprint_type="morgan_fingerprint",
            )
            _5000_molecules_4_process_time += (time() -
                                               start) / self.N_REPLICATES
        _5000_molecules_4_process_speedup = (
            self._5000_molecules_serial_time / _5000_molecules_4_process_time
        )
        _5000_molecules_4_process_efficiency = _5000_molecules_4_process_speedup / 4

        print("Running 5000 molecules with 8 processes.", flush=True)
        _5000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=8,
                fingerprint_type="morgan_fingerprint",
            )
            _5000_molecules_8_process_time += (time() -
                                               start) / self.N_REPLICATES
        _5000_molecules_8_process_speedup = (
            self._5000_molecules_serial_time / _5000_molecules_8_process_time
        )
        _5000_molecules_8_process_efficiency = _5000_molecules_8_process_speedup / 8

        # 10000 molecules
        print("Running 10000 molecules with 2 processes.", flush=True)
        _10000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=2,
                fingerprint_type="morgan_fingerprint",
            )
            _10000_molecules_2_process_time += (time() -
                                                start) / self.N_REPLICATES
        _10000_molecules_2_process_speedup = (
            self._10000_molecules_serial_time / _10000_molecules_2_process_time
        )
        _10000_molecules_2_process_efficiency = _10000_molecules_2_process_speedup / 2

        print("Running 10000 molecules with 4 processes.", flush=True)
        _10000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=4,
                fingerprint_type="morgan_fingerprint",
            )
            _10000_molecules_4_process_time += (time() -
                                                start) / self.N_REPLICATES
        _10000_molecules_4_process_speedup = (
            self._10000_molecules_serial_time / _10000_molecules_4_process_time
        )
        _10000_molecules_4_process_efficiency = _10000_molecules_4_process_speedup / 4

        print("Running 10000 molecules with 8 processes.", flush=True)
        _10000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=8,
                fingerprint_type="morgan_fingerprint",
            )
            _10000_molecules_8_process_time += (
                time() - start) / self.N_REPLICATES
        _10000_molecules_8_process_speedup = (
            self._10000_molecules_serial_time / _10000_molecules_8_process_time
        )
        _10000_molecules_8_process_efficiency = (
            _10000_molecules_8_process_speedup / 8
        )

        # 15000 molecules
        print("Running 15000 molecules with 2 processes.", flush=True)
        _15000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=2,
                fingerprint_type="morgan_fingerprint",
            )
            _15000_molecules_2_process_time += (time() -
                                                start) / self.N_REPLICATES
        _15000_molecules_2_process_speedup = (
            self._15000_molecules_serial_time / _15000_molecules_2_process_time
        )
        _15000_molecules_2_process_efficiency = _15000_molecules_2_process_speedup / 2

        print("Running 15000 molecules with 4 processes.", flush=True)
        _15000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=4,
                fingerprint_type="morgan_fingerprint",
            )
            _15000_molecules_4_process_time += (time() -
                                                start) / self.N_REPLICATES
        _15000_molecules_4_process_speedup = (
            self._15000_molecules_serial_time / _15000_molecules_4_process_time
        )
        _15000_molecules_4_process_efficiency = _15000_molecules_4_process_speedup / 4

        print("Running 15000 molecules with 8 processes.", flush=True)
        _15000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="tanimoto",
                n_threads=8,
                fingerprint_type="morgan_fingerprint",
            )
            _15000_molecules_8_process_time += (
                time() - start) / self.N_REPLICATES
        _15000_molecules_8_process_speedup = (
            self._15000_molecules_serial_time / _15000_molecules_8_process_time
        )
        _15000_molecules_8_process_efficiency = (
            _15000_molecules_8_process_speedup / 8
        )
        print("Speedup:", flush=True)
        print(
            tabulate(
                [
                    ["~", 2, 4, 8],
                    [
                        100,
                        _100_molecules_2_process_speedup,
                        _100_molecules_4_process_speedup,
                        _100_molecules_8_process_speedup,
                    ],
                    [
                        500,
                        _500_molecules_2_process_speedup,
                        _500_molecules_4_process_speedup,
                        _500_molecules_8_process_speedup,
                    ],
                    [
                        1000,
                        _1000_molecules_2_process_speedup,
                        _1000_molecules_4_process_speedup,
                        _1000_molecules_8_process_speedup,
                    ],
                    [
                        5000,
                        _5000_molecules_2_process_speedup,
                        _5000_molecules_4_process_speedup,
                        _5000_molecules_8_process_speedup,
                    ],
                    [
                        10000,
                        _10000_molecules_2_process_speedup,
                        _10000_molecules_4_process_speedup,
                        _10000_molecules_8_process_speedup,
                    ],
                    [
                        15000,
                        _15000_molecules_2_process_speedup,
                        _15000_molecules_4_process_speedup,
                        _15000_molecules_8_process_speedup,
                    ],
                ],
                headers=["# mol", "", "# processes", ""],
            )
        )
        print("Efficiency:", flush=True)
        print(
            tabulate(
                [
                    ["~", 2, 4, 8],
                    [
                        100,
                        _100_molecules_2_process_efficiency,
                        _100_molecules_4_process_efficiency,
                        _100_molecules_8_process_efficiency,
                    ],
                    [
                        500,
                        _500_molecules_2_process_efficiency,
                        _500_molecules_4_process_efficiency,
                        _500_molecules_8_process_efficiency,
                    ],
                    [
                        1000,
                        _1000_molecules_2_process_efficiency,
                        _1000_molecules_4_process_efficiency,
                        _1000_molecules_8_process_efficiency,
                    ],
                    [
                        5000,
                        _5000_molecules_2_process_efficiency,
                        _5000_molecules_4_process_efficiency,
                        _5000_molecules_8_process_efficiency,
                    ],
                    [
                        10000,
                        _10000_molecules_2_process_efficiency,
                        _10000_molecules_4_process_efficiency,
                        _10000_molecules_8_process_efficiency,
                    ],
                    [
                        15000,
                        _15000_molecules_2_process_efficiency,
                        _15000_molecules_4_process_efficiency,
                        _15000_molecules_8_process_efficiency,
                    ],
                ],
                headers=["# mol", "", "# processes", ""],
            )
        )
        print("Execution Time in seconds (serial/parallel):", flush=True)
        print(
            tabulate(
                [
                    ["~", 1, 2, 4, 8],
                    [
                        100,
                        "{:.2f}".format(
                            float(self._100_molecules_serial_time),
                        ),
                        "{:.2f}".format(
                            float(_100_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_100_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_100_molecules_8_process_time),
                        ),
                    ],
                    [
                        500,
                        "{:.2f}".format(
                            float(self._500_molecules_serial_time),
                        ),
                        "{:.2f}".format(
                            float(_500_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_500_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_500_molecules_8_process_time),
                        ),
                    ],
                    [
                        1000,
                        "{:.2f}".format(
                            float(self._1000_molecules_serial_time),
                        ),
                        "{:.2f}".format(
                            float(_1000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_1000_molecules_4_process_time),
                        ),
                        "{:2f}".format(
                            float(_1000_molecules_8_process_time),
                        ),
                    ],
                    [
                        5000,
                        "{:.2f}".format(
                            float(self._5000_molecules_serial_time),
                        ),
                        "{:.2f}".format(
                            float(_5000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_5000_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_5000_molecules_8_process_time),
                        ),
                    ],
                    [
                        10000,
                        "{:.2f}".format(
                            float(self._10000_molecules_serial_time),
                        ),
                        "{:.2f}".format(
                            float(_10000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_10000_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_10000_molecules_8_process_time),
                        ),
                    ],
                    [
                        15000,
                        "{:.2f}".format(
                            float(self._15000_molecules_serial_time),
                        ),
                        "{:.2f}".format(
                            float(_15000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_15000_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_15000_molecules_8_process_time),
                        ),
                    ],
                ],
                headers=["# mol", "", "", "# processes", ""],
            )
        )

    def test_speedup_efficiency_cosine(self):
        """
        Evaluate the speedup and efficieny of the multiprocessing approach
        with a more complex metric.
        """
        if self.NO_SPEEDUP_TEST:
            return
        print("~" * 10, "\n", "Speedup and Efficiency Test 2\n",
              "~" * 10, flush=True)
        # 100 molecules
        print("Running 100 molecules with 2 processes.", flush=True)
        _100_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=2,
                fingerprint_type="topological_fingerprint",
            )
            _100_molecules_2_process_time += (time() -
                                              start) / self.N_REPLICATES
        _100_molecules_2_process_speedup = (
            self._100_molecules_serial_time_2 / _100_molecules_2_process_time
        )
        _100_molecules_2_process_efficiency = _100_molecules_2_process_speedup / 2

        print("Running 100 molecules with 4 processes.", flush=True)
        _100_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=4,
                fingerprint_type="topological_fingerprint",
            )
            _100_molecules_4_process_time += (time() -
                                              start) / self.N_REPLICATES
        _100_molecules_4_process_speedup = (
            self._100_molecules_serial_time_2 / _100_molecules_4_process_time
        )
        _100_molecules_4_process_efficiency = _100_molecules_4_process_speedup / 4

        print("Running 100 molecules with 8 processes.", flush=True)
        _100_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._100_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=8,
                fingerprint_type="topological_fingerprint",
            )
            _100_molecules_8_process_time += (time() -
                                              start) / self.N_REPLICATES
        _100_molecules_8_process_speedup = (
            self._100_molecules_serial_time_2 / _100_molecules_8_process_time
        )
        _100_molecules_8_process_efficiency = _100_molecules_8_process_speedup / 8

        # 500 molecules
        print("Running 500 molecules with 2 processes.", flush=True)
        _500_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=2,
                fingerprint_type="topological_fingerprint",
            )
            _500_molecules_2_process_time += (time() -
                                              start) / self.N_REPLICATES
        _500_molecules_2_process_speedup = (
            self._500_molecules_serial_time_2 / _500_molecules_2_process_time
        )
        _500_molecules_2_process_efficiency = _500_molecules_2_process_speedup / 2

        print("Running 500 molecules with 4 processes.", flush=True)
        _500_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=4,
                fingerprint_type="topological_fingerprint",
            )
            _500_molecules_4_process_time += (time() -
                                              start) / self.N_REPLICATES
        _500_molecules_4_process_speedup = (
            self._500_molecules_serial_time_2 / _500_molecules_4_process_time
        )
        _500_molecules_4_process_efficiency = _500_molecules_4_process_speedup / 4

        print("Running 500 molecules with 8 processes.", flush=True)
        _500_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._500_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=8,
                fingerprint_type="topological_fingerprint",
            )
            _500_molecules_8_process_time += (time() -
                                              start) / self.N_REPLICATES
        _500_molecules_8_process_speedup = (
            self._500_molecules_serial_time_2 / _500_molecules_8_process_time
        )
        _500_molecules_8_process_efficiency = _500_molecules_8_process_speedup / 8

        # 1000 molecules
        print("Running 1000 molecules with 2 processes.", flush=True)
        _1000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=2,
                fingerprint_type="topological_fingerprint",
            )
            _1000_molecules_2_process_time += (time() -
                                               start) / self.N_REPLICATES
        _1000_molecules_2_process_speedup = (
            self._1000_molecules_serial_time_2 / _1000_molecules_2_process_time
        )
        _1000_molecules_2_process_efficiency = _1000_molecules_2_process_speedup / 2

        print("Running 1000 molecules with 4 processes.", flush=True)
        _1000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=4,
                fingerprint_type="topological_fingerprint",
            )
            _1000_molecules_4_process_time += (time() -
                                               start) / self.N_REPLICATES
        _1000_molecules_4_process_speedup = (
            self._1000_molecules_serial_time_2 / _1000_molecules_4_process_time
        )
        _1000_molecules_4_process_efficiency = _1000_molecules_4_process_speedup / 4

        print("Running 1000 molecules with 8 processes.", flush=True)
        _1000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._1000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=8,
                fingerprint_type="topological_fingerprint",
            )
            _1000_molecules_8_process_time += (time() -
                                               start) / self.N_REPLICATES
        _1000_molecules_8_process_speedup = (
            self._1000_molecules_serial_time_2 / _1000_molecules_8_process_time
        )
        _1000_molecules_8_process_efficiency = _1000_molecules_8_process_speedup / 8

        # 5000 molecules
        print("Running 5000 molecules with 2 processes.", flush=True)
        _5000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=2,
                fingerprint_type="topological_fingerprint",
            )
            _5000_molecules_2_process_time += (time() -
                                               start) / self.N_REPLICATES
        _5000_molecules_2_process_speedup = (
            self._5000_molecules_serial_time_2 / _5000_molecules_2_process_time
        )
        _5000_molecules_2_process_efficiency = _5000_molecules_2_process_speedup / 2

        print("Running 5000 molecules with 4 processes.", flush=True)
        _5000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=4,
                fingerprint_type="topological_fingerprint",
            )
            _5000_molecules_4_process_time += (time() -
                                               start) / self.N_REPLICATES
        _5000_molecules_4_process_speedup = (
            self._5000_molecules_serial_time_2 / _5000_molecules_4_process_time
        )
        _5000_molecules_4_process_efficiency = _5000_molecules_4_process_speedup / 4

        print("Running 5000 molecules with 8 processes.", flush=True)
        _5000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._5000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=8,
                fingerprint_type="topological_fingerprint",
            )
            _5000_molecules_8_process_time += (time() -
                                               start) / self.N_REPLICATES
        _5000_molecules_8_process_speedup = (
            self._5000_molecules_serial_time_2 / _5000_molecules_8_process_time
        )
        _5000_molecules_8_process_efficiency = _5000_molecules_8_process_speedup / 8

        # 10000 molecules
        print("Running 10000 molecules with 2 processes.", flush=True)
        _10000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=2,
                fingerprint_type="topological_fingerprint",
            )
            _10000_molecules_2_process_time += (time() -
                                                start) / self.N_REPLICATES
        _10000_molecules_2_process_speedup = (
            self._10000_molecules_serial_time_2 / _10000_molecules_2_process_time
        )
        _10000_molecules_2_process_efficiency = _10000_molecules_2_process_speedup / 2

        print("Running 10000 molecules with 4 processes.", flush=True)
        _10000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=4,
                fingerprint_type="topological_fingerprint",
            )
            _10000_molecules_4_process_time += (time() -
                                                start) / self.N_REPLICATES
        _10000_molecules_4_process_speedup = (
            self._10000_molecules_serial_time_2 / _10000_molecules_4_process_time
        )
        _10000_molecules_4_process_efficiency = _10000_molecules_4_process_speedup / 4

        print("Running 10000 molecules with 8 processes.", flush=True)
        _10000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._10000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=8,
                fingerprint_type="topological_fingerprint",
            )
            _10000_molecules_8_process_time += (
                time() - start) / self.N_REPLICATES
        _10000_molecules_8_process_speedup = (
            self._10000_molecules_serial_time_2 / _10000_molecules_8_process_time
        )
        _10000_molecules_8_process_efficiency = (
            _10000_molecules_8_process_speedup / 8
        )

        # 15000 molecules
        print("Running 15000 molecules with 2 processes.", flush=True)
        _15000_molecules_2_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=2,
                fingerprint_type="topological_fingerprint",
            )
            _15000_molecules_2_process_time += (time() -
                                                start) / self.N_REPLICATES
        _15000_molecules_2_process_speedup = (
            self._15000_molecules_serial_time_2 / _15000_molecules_2_process_time
        )
        _15000_molecules_2_process_efficiency = _15000_molecules_2_process_speedup / 2

        print("Running 15000 molecules with 4 processes.", flush=True)
        _15000_molecules_4_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=4,
                fingerprint_type="topological_fingerprint",
            )
            _15000_molecules_4_process_time += (time() -
                                                start) / self.N_REPLICATES
        _15000_molecules_4_process_speedup = (
            self._15000_molecules_serial_time_2 / _15000_molecules_4_process_time
        )
        _15000_molecules_4_process_efficiency = _15000_molecules_4_process_speedup / 4

        print("Running 15000 molecules with 8 processes.", flush=True)
        _15000_molecules_8_process_time = 0
        for i in range(self.N_REPLICATES):
            start = time()
            test_molecule_set = MoleculeSet(
                molecule_database_src=self._15000_molecules_fpath,
                molecule_database_src_type="text",
                is_verbose=False,
                similarity_measure="cosine",
                n_threads=8,
                fingerprint_type="topological_fingerprint",
            )
            _15000_molecules_8_process_time += (
                time() - start) / self.N_REPLICATES
        _15000_molecules_8_process_speedup = (
            self._15000_molecules_serial_time_2 / _15000_molecules_8_process_time
        )
        _15000_molecules_8_process_efficiency = (
            _15000_molecules_8_process_speedup / 8
        )
        print("Speedup:", flush=True)
        print(
            tabulate(
                [
                    ["~", 2, 4, 8],
                    [
                        100,
                        _100_molecules_2_process_speedup,
                        _100_molecules_4_process_speedup,
                        _100_molecules_8_process_speedup,
                    ],
                    [
                        500,
                        _500_molecules_2_process_speedup,
                        _500_molecules_4_process_speedup,
                        _500_molecules_8_process_speedup,
                    ],
                    [
                        1000,
                        _1000_molecules_2_process_speedup,
                        _1000_molecules_4_process_speedup,
                        _1000_molecules_8_process_speedup,
                    ],
                    [
                        5000,
                        _5000_molecules_2_process_speedup,
                        _5000_molecules_4_process_speedup,
                        _5000_molecules_8_process_speedup,
                    ],
                    [
                        10000,
                        _10000_molecules_2_process_speedup,
                        _10000_molecules_4_process_speedup,
                        _10000_molecules_8_process_speedup,
                    ],
                    [
                        15000,
                        _15000_molecules_2_process_speedup,
                        _15000_molecules_4_process_speedup,
                        _15000_molecules_8_process_speedup,
                    ],
                ],
                headers=["# mol", "", "# processes", ""],
            )
        )
        print("Efficiency:", flush=True)
        print(
            tabulate(
                [
                    ["~", 2, 4, 8],
                    [
                        100,
                        _100_molecules_2_process_efficiency,
                        _100_molecules_4_process_efficiency,
                        _100_molecules_8_process_efficiency,
                    ],
                    [
                        500,
                        _500_molecules_2_process_efficiency,
                        _500_molecules_4_process_efficiency,
                        _500_molecules_8_process_efficiency,
                    ],
                    [
                        1000,
                        _1000_molecules_2_process_efficiency,
                        _1000_molecules_4_process_efficiency,
                        _1000_molecules_8_process_efficiency,
                    ],
                    [
                        5000,
                        _5000_molecules_2_process_efficiency,
                        _5000_molecules_4_process_efficiency,
                        _5000_molecules_8_process_efficiency,
                    ],
                    [
                        10000,
                        _10000_molecules_2_process_efficiency,
                        _10000_molecules_4_process_efficiency,
                        _10000_molecules_8_process_efficiency,
                    ],
                    [
                        15000,
                        _15000_molecules_2_process_efficiency,
                        _15000_molecules_4_process_efficiency,
                        _15000_molecules_8_process_efficiency,
                    ],
                ],
                headers=["# mol", "", "# processes", ""],
            )
        )
        print("Execution Time in seconds (serial/parallel):", flush=True)
        print(
            tabulate(
                [
                    ["~", 1, 2, 4, 8],
                    [
                        100,
                        "{:.2f}".format(
                            float(self._100_molecules_serial_time_2),
                        ),
                        "{:.2f}".format(
                            float(_100_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_100_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_100_molecules_8_process_time),
                        ),
                    ],
                    [
                        500,
                        "{:.2f}".format(
                            float(self._500_molecules_serial_time_2),
                        ),
                        "{:.2f}".format(
                            float(_500_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_500_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_500_molecules_8_process_time),
                        ),
                    ],
                    [
                        1000,
                        "{:.2f}".format(
                            float(self._1000_molecules_serial_time_2),
                        ),
                        "{:.2f}".format(
                            float(_1000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_1000_molecules_4_process_time),
                        ),
                        "{:2f}".format(
                            float(_1000_molecules_8_process_time),
                        ),
                    ],
                    [
                        5000,
                        "{:.2f}".format(
                            float(self._5000_molecules_serial_time_2),
                        ),
                        "{:.2f}".format(
                            float(_5000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_5000_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_5000_molecules_8_process_time),
                        ),
                    ],
                    [
                        10000,
                        "{:.2f}".format(
                            float(self._10000_molecules_serial_time_2),
                        ),
                        "{:.2f}".format(
                            float(_10000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_10000_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_10000_molecules_8_process_time),
                        ),
                    ],
                    [
                        15000,
                        "{:.2f}".format(
                            float(self._15000_molecules_serial_time_2),
                        ),
                        "{:.2f}".format(
                            float(_15000_molecules_2_process_time),
                        ),
                        "{:.2f}".format(
                            float(_15000_molecules_4_process_time),
                        ),
                        "{:.2f}".format(
                            float(_15000_molecules_8_process_time),
                        ),
                    ],
                ],
                headers=["# mol", "", "", "# processes", ""],
            )
        )

    @classmethod
    def tearDownClass(self):
        """Delete temporary files used in testing."""
        print("Deleting smiles database files.", flush=True)
        remove(self.text_fpath)
        if not self.NO_SPEEDUP_TEST:
            remove(self._100_molecules_fpath)
            remove(self._500_molecules_fpath)
            remove(self._1000_molecules_fpath)
            remove(self._5000_molecules_fpath)
            remove(self._10000_molecules_fpath)
            remove(self._15000_molecules_fpath)
        print(" ~ ~ Multithreading Test Complete ~ ~ ", flush=True)


if __name__ == "__main__":
    unittest.main()
