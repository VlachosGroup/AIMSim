""" Test the task CompareTargetMolecule  """
import unittest
from os import remove

import numpy as np
import pandas as pd

from aimsim.chemical_datastructures import Molecule, MoleculeSet
from aimsim.ops import Descriptor, SimilarityMeasure
from aimsim.tasks import CompareTargetMolecule


SUPPORTED_SIMILARITIES = SimilarityMeasure.get_supported_metrics()
SUPPORTED_FPRINTS = Descriptor.get_supported_fprints()

TEST_SMILES = [
    "CCCCCCC",
    "CCCC",
    "CCC",
    "CO",
    "CN",
    "C1=CC=CC=C1",
    "CC1=CC=CC=C1",
    "C(=O)(N)N",
]


class TestCompareTargetMolecule(unittest.TestCase):

    def smiles_seq_to_xl_or_csv(
        self, ftype, property_seq=None, name_seq=None, feature_arr=None
    ):
        """Helper method to convert a SMILES sequence or arbitrary features
        to Excel or CSV files.

        Args:
            ftype (str): String label to denote the filetype. 'csv' or 'excel'.
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses. Defaults to None.
            name_seq (list or np.ndarray, optional): Optional sequence of
                molecular names. Defaults to None.
            feature_arr (np.ndarray, optional): Optional array of molecular
                descriptor values. Defaults to None.

        Raises:
            ValueError: Invalid file type specified.

        Returns:
            str: Path to created file.
        """
        data = {"descriptor_smiles": TEST_SMILES}
        if property_seq is not None:
            data.update({"response_random": property_seq})
        if name_seq is not None:
            data.update({"descriptor_name": name_seq})
        if feature_arr is not None:
            feature_arr = np.array(feature_arr)
            for feature_num in range(feature_arr.shape[1]):
                data.update({f"descriptor_{feature_num}":
                            feature_arr[:, feature_num]})
        data_df = pd.DataFrame(data)
        fpath = "temp_mol_file"
        if ftype == "excel":
            fpath += ".xlsx"
            print(f"Creating {ftype} file {fpath}")
            data_df.to_excel(fpath)
        elif ftype == "csv":
            fpath += ".csv"
            print(f"Creating {ftype} file {fpath}")
            data_df.to_csv(fpath)
        else:
            raise ValueError(f"{ftype} not supported")
        return fpath

    def test_get_molecule_most_similar_to(self):
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv")
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type="csv",
                    fingerprint_type=descriptor,
                    similarity_measure=similarity_measure,
                    is_verbose=False,
                )
                for mol_smile, mol in zip(TEST_SMILES,
                                          molecule_set.molecule_database):
                    compare_task = CompareTargetMolecule(
                        target_molecule_smiles=mol_smile)
                    [closest_mol], [similarity] = compare_task.\
                        get_hits_similar_to(molecule_set)
                    mol_similarities = molecule_set.compare_against_molecule(
                        mol)
                    self.assertEqual(
                        np.max(mol_similarities),
                        mol.get_similarity_to(
                            molecule_set.molecule_database[closest_mol],
                            molecule_set.similarity_measure
                        ),
                        f"Expected closest mol to have maximum "
                        f"similarity to target molecule "
                        f"using similarity measure: "
                        f"{similarity_measure}, "
                        f"descriptor: "
                        f"{descriptor}, "
                        f"for molecule {mol.mol_text}",
                    )
                    self.assertGreaterEqual(similarity, 0.,
                                            "Expected similarity value to "
                                            "be >= 0."
                                            f"using similarity measure: "
                                            f"{similarity_measure}, "
                                            f"descriptor: {descriptor}, "
                                            f"for molecule {mol.mol_text}")
                    self.assertLessEqual(similarity, 1.,
                                         "Expected similarity value to "
                                         "be <= 1."
                                         f"using similarity measure: "
                                         f"{similarity_measure}, "
                                         f"descriptor: {descriptor}, "
                                         f"for molecule {mol.mol_text}"
                                         )

    def test_get_molecule_least_similar_to(self):
        """Test for get_molecule_least_similar_to functionality."""
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv")
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type="csv",
                    fingerprint_type=descriptor,
                    similarity_measure=similarity_measure,
                    is_verbose=False,
                )
                for mol_smile, mol in zip(TEST_SMILES,
                                          molecule_set.molecule_database):
                    compare_task = CompareTargetMolecule(
                        target_molecule_smiles=mol_smile)
                    [furthest_mol], [similarity] = compare_task.\
                        get_hits_dissimilar_to(molecule_set)
                    mol_similarities = molecule_set.compare_against_molecule(
                        mol)
                    self.assertEqual(
                        np.min(mol_similarities),
                        mol.get_similarity_to(
                            molecule_set.molecule_database[furthest_mol],
                            molecule_set.similarity_measure
                        ),
                        f"Expected furthest mol to have minimum "
                        f"similarity to target molecule "
                        f"using similarity measure: {similarity_measure}, "
                        f"descriptor: {descriptor}, "
                        f"for molecule {mol.mol_text}",
                    )
                    self.assertGreaterEqual(similarity,  0.,
                                            "Expected similarity value to "
                                            "be >= 0."
                                            f"using similarity measure: "
                                            f"{similarity_measure}, "
                                            f"descriptor: {descriptor}, "
                                            f"for molecule {mol.mol_text}")
                    self.assertLessEqual(similarity, 1.,
                                         "Expected similarity value to "
                                         "be <= 1."
                                         f"using similarity measure: "
                                         f"{similarity_measure}, "
                                         f"descriptor: {descriptor}, "
                                         f"for molecule {mol.mol_text}"
                                         )

    @classmethod
    def tearDownClass(self):
        """Delete temporary file used in testing."""
        remove('temp_mol_file.csv')


if __name__ == "__main__":
    unittest.main()
