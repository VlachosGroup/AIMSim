"""Test the SimilarityMEasure class"""
from os import remove
import unittest

import pandas as pd

from aimsim.chemical_datastructures import Molecule, MoleculeSet
from aimsim.ops import SimilarityMeasure, Descriptor


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


class TestSimilarityMeasure(unittest.TestCase):

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

    def test_all_supported_measures(self):
        supported_measures = SimilarityMeasure.get_supported_metrics()
        for measure in supported_measures:
            try:
                _ = SimilarityMeasure(metric=measure)
            except ValueError:
                self.fail(f'Did not expect {measure} similarity metric to '
                          f'raise ValueError')

    def test_get_abcd(self):
        similarity_measure = SimilarityMeasure('tanimoto')

        def _check_abcd(true_vals, arr1, arr2):
            fp1 = Descriptor(arr1)
            fp1.label_ = 'arbitrary_fingerprint'
            fp2 = Descriptor(arr2)
            fp2.label_ = 'arbitrary_fingerprint'
            abcd_calc = similarity_measure._get_abcd(fp1, fp2)
            for var_id, var in enumerate(['a', 'b', 'c', 'd']):
                self.assertEqual(true_vals[var], abcd_calc[var_id],
                                 f'Expected true {var} to match calculated val '
                                 f'for arrays {arr1}, {arr2}')

        # Case 1
        arr1 = [1, 1, 1, 1, 1]
        arr2 = [0, 0, 0, 0, 0]
        true_vals = {'a': 0, 'b': 5, 'c': 0, 'd': 0}
        _check_abcd(true_vals, arr1=arr1, arr2=arr2)

        # Case 2
        arr1 = [1, 1, 1, 0]
        arr2 = [0, 1]
        true_vals = {'a': 1, 'b': 1, 'c': 0, 'd': 0}
        _check_abcd(true_vals, arr1=arr1, arr2=arr2)

        # Case 3
        arr1 = [1, 0, 1, 0]
        arr2 = [1, 0, 1, 0]
        true_vals = {'a': 2, 'b': 0, 'c': 0, 'd': 2}
        _check_abcd(true_vals, arr1=arr1, arr2=arr2)

        # Case 4
        arr1 = [0, 1, 0, 1]
        arr2 = [1, 0, 1, 0]
        true_vals = {'a': 0, 'b': 2, 'c': 2, 'd': 0}
        _check_abcd(true_vals, arr1=arr1, arr2=arr2)

        # Case 5
        arr1 = [1, 0, 0, 1, 1]
        arr2 = [1, 0, 1, 0, 0]
        true_vals = {'a': 1, 'b': 2, 'c': 1, 'd': 1}
        _check_abcd(true_vals, arr1=arr1, arr2=arr2)

    def test_similarity_measure_limits(self):
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
                for mol1 in molecule_set.molecule_database:
                    for mol2 in molecule_set.molecule_database:
                        similarity_ = mol1.get_similarity_to(
                            mol2,
                            molecule_set.similarity_measure)
                        self.assertGreaterEqual(similarity_, 0.,
                                                "Expected similarity value "
                                                "to be >= 0."
                                                f"using similarity measure:"
                                                f" {similarity_measure}, "
                                                f"descriptor: {descriptor}"
                                                f", for molecules "
                                                f"{mol1.mol_text}, "
                                                f"{mol2.mol_text}")
                        self.assertLessEqual(similarity_, 1.,
                                             "Expected similarity value to "
                                             "be <= 1."
                                             f"using similarity measure: "
                                             f"{similarity_measure}, "
                                             f"descriptor: {descriptor}, "
                                             f"for molecule {mol1.mol_text}, "
                                             f"{mol2.mol_text}"
                                             )
        remove('temp_mol_file.csv')


if __name__ == "__main__":
    unittest.main()
