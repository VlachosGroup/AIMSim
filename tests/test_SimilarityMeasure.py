"""Test the SimilarityMEasure class"""
import unittest

from molSim.ops import SimilarityMeasure, Descriptor


class TestSimilarityMeasure(unittest.TestCase):
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









