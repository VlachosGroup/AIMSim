"""Test for the Descriptor class"""
import unittest
import numpy as np
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import MolFromSmiles
from molSim.ops import Descriptor

SUPPORTED_FPRINTS = Descriptor.get_supported_fprints()


class TestDescriptor(unittest.TestCase):
    def test_descriptor_empty_init(self):
        descriptor = Descriptor()
        self.assertFalse(descriptor.check_init(),
                         'Expected Descriptor object to be uninitialized')

    def test_descriptor_arbitrary_list_init(self):
        descriptor_value = [1, 2, 3]
        descriptor = Descriptor(value=descriptor_value)
        self.assertTrue(descriptor.check_init(),
                        'Expected Descriptor object to be initialized')
        self.assertEqual(descriptor.label_, 'arbitrary',
                         'Expected label of descriptor initialized with '
                         'arbitrary vector to be "arbitrary"')
        self.assertIsInstance(descriptor.to_numpy(), np.ndarray,
                              'Expected numpy.ndarray value from to_numpy()')
        self.assertEqual(descriptor.to_numpy().tolist(), descriptor_value,
                         'Expected descriptor value to match init value')
        with self.assertRaises(ValueError):
            descriptor.to_rdkit()

    def test_descriptor_arbitrary_numpy_init(self):
        descriptor_value = np.array([1, 2, 3])
        descriptor = Descriptor(value=descriptor_value)
        self.assertTrue(descriptor.check_init(),
                        'Expected Descriptor object to be initialized')
        self.assertEqual(descriptor.label_, 'arbitrary',
                         'Expected label of descriptor initialized with '
                         'arbitrary vector to be "arbitrary"')
        self.assertIsInstance(descriptor.to_numpy(), np.ndarray,
                              'Expected numpy.ndarray from to_numpy()')
        self.assertTrue((descriptor.to_numpy() == descriptor_value).all(),
                        'Expected descriptor value to match init value')
        with self.assertRaises(ValueError):
            descriptor.to_rdkit()

    def test_descriptor_make_fingerprint(self):
        mol_graph = MolFromSmiles('CCC')
        for fprint in SUPPORTED_FPRINTS:
            descriptor = Descriptor()
            descriptor.make_fingerprint(molecule_graph=mol_graph,
                                        fingerprint_type=fprint)
            self.assertTrue(descriptor.check_init(),
                            'Expected Descriptor object to be initialized')
            self.assertEqual(descriptor.label_, fprint,
                             'Expected label of descriptor initialized with '
                             'fingerprint to match the fingerprint')
            self.assertIsInstance(descriptor.to_numpy(), np.ndarray,
                                  'Expected numpy.ndarray from to_numpy()')
            self.assertIsInstance(descriptor.to_rdkit(), ExplicitBitVect,
                                  'Expected to_rdkit() to return '
                                  'ExplicitBitVect representation '
                                  f'of {fprint} fingerprint')
