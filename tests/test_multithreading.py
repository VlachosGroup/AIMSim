""" Test multithreading to ensure consistent behavior with serial implementation."""
import unittest
from os import remove
import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import MolFromSmiles
from molSim.chemical_datastructures import Molecule, MoleculeSet
from molSim.ops import Descriptor, SimilarityMeasure
from molSim.tasks.visualize_dataset import VisualizeDataset


class TestMultithreading(unittest.TestCase):
    """Unit tests to ensure consistency when running molSim as a single process
    or when using multiprocessing.
    """
    @classmethod
    def setUpClass(self):
        """Create a SMILES database to use for comparisons and find the similarity matrix.
        """
        print(" ~ ~ Testing Multithreading ~ ~ ")
        self.text_fpath = 'temp_multithread_smiles_seq.txt'
        print(f'Creating text file {self.text_fpath}')
        with open(self.text_fpath, "w") as file:
            for smiles in ['C', 'CC', 'CCC', 'O', 'CCCC', 'CO', 'CCOCC']:
                file.write(smiles + '\n')
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure='tanimoto',
            n_threads=1,
            fingerprint_type='morgan_fingerprint',
        )
        self.correct_similarity_matrix = test_molecule_set.get_similarity_matrix()

    def test_multithreading_consistency_2_threads(self):
        """
        Ensure that the similarity matrix produced with 2 threads is identical to
        that produced using a single thread and the serial implementation.
        """
        test_molecule_set = MoleculeSet(
            molecule_database_src=self.text_fpath,
            molecule_database_src_type="text",
            is_verbose=True,
            similarity_measure='tanimoto',
            n_threads=2,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using two threads."
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
            similarity_measure='tanimoto',
            n_threads=3,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using three threads."
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
            similarity_measure='tanimoto',
            n_threads=4,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using four threads."
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
            similarity_measure='tanimoto',
            n_threads=5,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using five threads."
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
            similarity_measure='tanimoto',
            n_threads=6,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using six threads."
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
            similarity_measure='tanimoto',
            n_threads=7,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using seven threads (equal to the number of molecules)."
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
            similarity_measure='tanimoto',
            n_threads=10,
            fingerprint_type='morgan_fingerprint',
        )
        self.assertIsNone(
            np.testing.assert_array_equal(test_molecule_set.get_similarity_matrix(),
                                          self.correct_similarity_matrix),
            "Similarity matrix not equal when using ten threads (more than the number of molecules)."
        )

    @classmethod
    def tearDownClass(self):
        print('Deleting smiles database file.')
        remove(self.text_fpath)
        print(" ~ ~ Multithreading Test Complete ~ ~ ")


if __name__ == '__main__':
    unittest.main()
