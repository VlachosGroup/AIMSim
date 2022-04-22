""" Test the ValueError exceptions in SimilarityMeasure """
import unittest

from aimsim.ops.similarity_measures import SimilarityMeasure
from aimsim.ops.descriptor import Descriptor


class TestSimilarityMeasureValueError(unittest.TestCase):
    """
    Tests for Value Error exception in SimilarityMeasure

    """

    def test_invalid_metric(self):
        """Initializing with a non-existent metric should throw an error.
        """
        with self.assertRaises(ValueError):
            sim_mes = SimilarityMeasure("this metric does not exist")

    def test_empty_fprints(self):
        """Empty fingerprints cannot be compared.
        """
        desc_1 = Descriptor([0])
        desc_2 = Descriptor([0])
        sim_mes = SimilarityMeasure("tanimoto")
        with self.assertRaises(ValueError):
            sim_mes(desc_1, desc_2)

    def test_vectornorm_length_errors(self):
        """
        Vector norm-based similarities should only work with
        descriptors of the same length, otherwise it should raise
        a value error.
        """
        desc_1 = Descriptor([1, 2])
        desc_2 = Descriptor([3])
        sim_mes_1 = SimilarityMeasure("l0_similarity")
        sim_mes_2 = SimilarityMeasure("l1_similarity")
        sim_mes_3 = SimilarityMeasure("l2_similarity")
        with self.assertRaises(ValueError):
            sim_mes_1(desc_1, desc_2)
        with self.assertRaises(ValueError):
            sim_mes_2(desc_1, desc_2)
        with self.assertRaises(ValueError):
            sim_mes_3(desc_1, desc_2)

    def test_compatible_metrics(self):
        """Test get_compatible_metrics basic functionailty.
        """
        self.assertIsInstance(
            SimilarityMeasure.get_compatible_metrics(),
            dict,
        )

    def test_binary_only_metrics(self):
        """Metrics which only allow bit vectors should raise
        a value error on other inputs.
        """
        desc_1 = Descriptor([1, 2])
        desc_2 = Descriptor([3, 4])
        for metric in SimilarityMeasure.get_supported_binary_metrics():
            sim_mes = SimilarityMeasure(metric)
            with self.assertRaises(ValueError):
                sim_mes(desc_1, desc_2)


if __name__ == "__main__":
    unittest.main()
