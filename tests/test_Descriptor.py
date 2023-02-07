"""Tests for the Descriptor class"""
import unittest
import numpy as np
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import MolFromSmiles
from aimsim.ops import Descriptor
from aimsim.exceptions import MordredCalculatorError, InvalidConfigurationError
from aimsim.utils.extras import requires_mordred

SUPPORTED_FPRINTS = Descriptor.get_supported_fprints()


class TestDescriptor(unittest.TestCase):
    """Tests for methods of the Descriptor class.

    Args:
        unittest (unittest): Python unittest module.
    """

    def test_descriptor_empty_init(self):
        """
        Test to verify empty Descriptor object can be created.

        """
        descriptor = Descriptor()
        self.assertFalse(
            descriptor.check_init(),
            "Expected Descriptor object to be uninitialized",
        )

    def test_descriptor_arbitrary_list_init(self):
        """
        Test to verify creation of Descriptor object initialized
        by arbitrary list.

        """
        descriptor_value = [1, 2, 3]
        descriptor = Descriptor(value=descriptor_value)
        self.assertTrue(
            descriptor.check_init(),
            "Expected Descriptor object to be initialized",
        )
        self.assertEqual(
            descriptor.label_,
            "arbitrary",
            "Expected label of descriptor initialized with "
            'arbitrary vector to be "arbitrary"',
        )
        self.assertIsInstance(
            descriptor.to_numpy(),
            np.ndarray,
            "Expected numpy.ndarray value from to_numpy()",
        )
        self.assertEqual(
            descriptor.to_numpy().tolist(),
            descriptor_value,
            "Expected descriptor value to match init value",
        )
        with self.assertRaises(ValueError):
            descriptor.to_rdkit()

    def test_descriptor_arbitrary_numpy_init(self):
        """
        Test to verify creation of Descriptor object initialized
        by arbitrary numpy array.

        """
        descriptor_value = np.array([1, 2, 3])
        descriptor = Descriptor(value=descriptor_value)
        self.assertTrue(
            descriptor.check_init(),
            "Expected Descriptor object to be initialized",
        )
        self.assertEqual(
            descriptor.label_,
            "arbitrary",
            "Expected label of descriptor initialized with "
            'arbitrary vector to be "arbitrary"',
        )
        self.assertIsInstance(
            descriptor.to_numpy(), np.ndarray, "Expected numpy.ndarray from to_numpy()"
        )
        self.assertTrue(
            (descriptor.to_numpy() == descriptor_value).all(),
            "Expected descriptor value to match init value",
        )
        with self.assertRaises(ValueError):
            descriptor.to_rdkit()

    def test_topological_fprint_min_path_lesser_than_atoms(self):
        atomic_mols = [MolFromSmiles(smiles) for smiles in ["C", "O", "N", "P"]]
        diatomic_mols = [MolFromSmiles(smiles) for smiles in ["CC", "CO", "CN", "CP"]]
        triatomic_mols = [
            MolFromSmiles(smiles) for smiles in ["CCC", "COO", "CCN", "CCP"]
        ]
        min_path = 1
        for mol in atomic_mols:
            with self.assertRaises(InvalidConfigurationError):
                descriptor = Descriptor()
                descriptor.make_fingerprint(
                    molecule_graph=mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
        for diatomic_mol in diatomic_mols:
            descriptor = Descriptor()
            try:
                descriptor.make_fingerprint(
                    molecule_graph=diatomic_mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
            except InvalidConfigurationError:
                self.fail(
                    "Did not expect Descriptor to raise " "InvalidConfigurationError"
                )
        for triatomic_mol in triatomic_mols:
            descriptor = Descriptor()
            try:
                descriptor.make_fingerprint(
                    molecule_graph=triatomic_mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
            except InvalidConfigurationError:
                self.fail(
                    "Did not expect Descriptor to raise " "InvalidConfigurationError"
                )

        min_path = 2
        for mol in atomic_mols:
            with self.assertRaises(InvalidConfigurationError):
                descriptor = Descriptor()
                descriptor.make_fingerprint(
                    molecule_graph=mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
        for diatomic_mol in diatomic_mols:
            with self.assertRaises(InvalidConfigurationError):
                descriptor = Descriptor()
                descriptor.make_fingerprint(
                    molecule_graph=diatomic_mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
        for triatomic_mol in triatomic_mols:
            descriptor = Descriptor()
            try:
                descriptor.make_fingerprint(
                    molecule_graph=triatomic_mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
            except InvalidConfigurationError:
                self.fail(
                    "Did not expect Descriptor to raise " "InvalidConfigurationError"
                )

        min_path = 3
        for mol in atomic_mols:
            with self.assertRaises(InvalidConfigurationError):
                descriptor = Descriptor()
                descriptor.make_fingerprint(
                    molecule_graph=mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
        for diatomic_mol in diatomic_mols:
            with self.assertRaises(InvalidConfigurationError):
                descriptor = Descriptor()
                descriptor.make_fingerprint(
                    molecule_graph=diatomic_mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )
        for triatomic_mol in triatomic_mols:
            with self.assertRaises(InvalidConfigurationError):
                descriptor = Descriptor()
                descriptor.make_fingerprint(
                    molecule_graph=triatomic_mol,
                    fingerprint_type="topological_fingerprint",
                    fingerprint_params={"min_path": min_path},
                )

    def test_descriptor_make_fingerprint(self):
        """
        Test to verify creation of Descriptor object by
        creating molecular fingerprints from the molecule graph.

        """
        mol_graph = MolFromSmiles("CCC")
        for fprint in SUPPORTED_FPRINTS:
            descriptor = Descriptor()
            descriptor.make_fingerprint(
                molecule_graph=mol_graph, fingerprint_type=fprint
            )
            self.assertTrue(
                descriptor.check_init(),
                "Expected Descriptor object to be initialized",
            )
            self.assertEqual(
                descriptor.label_,
                fprint,
                "Expected label of descriptor initialized with "
                "fingerprint to match the fingerprint",
            )
            self.assertIsInstance(
                descriptor.to_numpy(),
                np.ndarray,
                "Expected numpy.ndarray from to_numpy()",
            )
            self.assertIsInstance(
                descriptor.to_rdkit(),
                ExplicitBitVect,
                "Expected to_rdkit() to return "
                "ExplicitBitVect representation "
                f"of {fprint} fingerprint",
            )

    @requires_mordred
    def test_mordred_descriptors(self):
        """Test ability to passthrough descriptors to Mordred."""
        mol_graph = MolFromSmiles(
            "CC(C)C1=CC(=C(C(=C1)C(C)C)C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C)C"
        )
        for desc in ["MW", "LogEE_Dt", "BalabanJ"]:
            descriptor = Descriptor()
            descriptor.make_fingerprint(
                molecule_graph=mol_graph, fingerprint_type="mordred:" + desc
            )
            self.assertTrue(
                descriptor.check_init(),
                "Expected Descriptor object to be initialized",
            )
            self.assertEqual(
                descriptor.label_,
                desc,
                "Expected label of descriptor initialized with "
                "{} to match the fingerprint".format(desc),
            )
            self.assertIsInstance(
                descriptor.to_numpy(),
                np.ndarray,
                "Expected numpy.ndarray from to_numpy()",
            )
            with self.assertRaises(ValueError):
                descriptor.to_rdkit()

    def test_padelpy_descriptors(self):
        """Test ability to passthrough descriptors to PadelPy."""
        mol_graph = MolFromSmiles("CCOCC")
        for desc in ["MATS7e", "Ti", "ATSC6p"]:
            descriptor = Descriptor()
            descriptor.make_fingerprint(
                molecule_graph=mol_graph, fingerprint_type="padelpy:" + desc
            )
            self.assertTrue(
                descriptor.check_init(),
                "Expected Descriptor object to be initialized",
            )
            self.assertEqual(
                descriptor.label_,
                desc,
                "Expected label of descriptor initialized with "
                "{} to match the fingerprint".format(desc),
            )
            self.assertIsInstance(
                descriptor.to_numpy(),
                np.ndarray,
                "Expected numpy.ndarray from to_numpy()",
            )
            with self.assertRaises(ValueError):
                descriptor.to_rdkit()

    def test_ccbmlib_descriptors(self):
        """Test ability to passthrough descriptors to ccbmlib."""
        mol_graph = MolFromSmiles("CCOCC")
        fprint_list = [
            "atom_pairs",
            "hashed_atom_pairs",
            "avalon",
            "maccs_keys",
            "morgan",
            "hashed_morgan",
            "rdkit_fingerprint",
            "torsions",
            "hashed_torsions",
        ]
        for desc in fprint_list:
            descriptor = Descriptor()
            descriptor.make_fingerprint(
                molecule_graph=mol_graph, fingerprint_type="ccbmlib:" + desc
            )
            self.assertTrue(
                descriptor.check_init(),
                "Expected Descriptor object to be initialized",
            )
            self.assertEqual(
                descriptor.label_,
                desc,
                "Expected label of descriptor initialized with "
                "{} to match the fingerprint".format(desc),
            )

    def test_exptl_descriptors(self):
        """Test ability to use experimental descriptors."""
        mol_graph = MolFromSmiles("CCOCC")
        fprint_list = [
            "maccs_keys",
            "atom-pair_fingerprint",
            "torsion_fingerprint",
        ]
        for desc in fprint_list:
            descriptor = Descriptor()
            descriptor.make_fingerprint(molecule_graph=mol_graph, fingerprint_type=desc)
            self.assertTrue(
                descriptor.check_init(),
                "Expected Descriptor object to be initialized",
            )
            self.assertEqual(
                descriptor.label_,
                desc,
                "Expected label of descriptor initialized with "
                "{} to match the fingerprint".format(desc),
            )

    @requires_mordred
    def test_nonexistent_mordred_descriptors(self):
        """Test ability to pass through descriptors to Mordred."""
        mol_graph = MolFromSmiles("C")
        for desc in ["", "ReallyInvalidDescriptorName"]:
            descriptor = Descriptor()
            with self.assertRaises(MordredCalculatorError):
                descriptor.make_fingerprint(
                    molecule_graph=mol_graph,
                    fingerprint_type="mordred:" + desc,
                )

    def test_bad_descriptors_padelpy_descriptors(self):
        """Test ability to pass through invalid descriptors to padelpy."""
        mol_graph = MolFromSmiles("C")
        for desc in ["", "ReallyInvalidDescriptorName"]:
            descriptor = Descriptor()
            with self.assertRaises(RuntimeError):
                descriptor.make_fingerprint(
                    molecule_graph=mol_graph,
                    fingerprint_type="padelpy:" + desc,
                    fingerprint_params={"timeout": 2},
                )

    def test_fingerprint_folding(self):
        """Create arbitrary fingerprint vector to check fold method"""
        # Case 1
        arbit_vector = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        arbit_label = "arbitrary"
        desc = Descriptor()
        desc.label_ = arbit_label
        desc.numpy_ = arbit_vector
        with self.assertRaises(ValueError):
            desc.get_folded_fprint(fold_to_length=2)

        # Case 2
        arbit_vector = np.array([1, 0, 1, 0, 1, 0])
        folded_vector = np.array([1, 1, 1])
        arbit_label = "arbitrary_fingerprint"
        desc = Descriptor()
        desc.label_ = arbit_label
        desc.numpy_ = arbit_vector
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=4)
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=10)
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=3) == folded_vector).all())
        )

        # Case 3
        arbit_vector = np.array([1, 0, 1, 0, 0, 0, 0, 0])
        folded_once_vector = np.array([1, 0, 1, 0])
        folded_twice_vector = np.array([1, 0])
        arbit_label = "arbitrary_fingerprint"
        desc = Descriptor()
        desc.label_ = arbit_label
        desc.numpy_ = arbit_vector
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=3)
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=10)
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=4) == folded_once_vector).all())
        )
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=2) == folded_twice_vector).all())
        )

        # Case 3
        arbit_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        folded_once_vector = np.array([0, 0, 0, 0])
        folded_twice_vector = np.array([0, 0])
        arbit_label = "arbitrary_fingerprint"
        desc = Descriptor()
        desc.label_ = arbit_label
        desc.numpy_ = arbit_vector
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=3)
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=10)
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=4) == folded_once_vector).all())
        )
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=2) == folded_twice_vector).all())
        )

        # Case 4
        arbit_vector = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        folded_once_vector = np.array([1, 1, 1, 1])
        folded_twice_vector = np.array([1, 1])
        arbit_label = "arbitrary_fingerprint"
        desc = Descriptor()
        desc.label_ = arbit_label
        desc.numpy_ = arbit_vector
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=3)
        with self.assertRaises(InvalidConfigurationError):
            desc.get_folded_fprint(fold_to_length=10)
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=4) == folded_once_vector).all())
        )
        self.assertTrue(
            ((desc.get_folded_fprint(fold_to_length=2) == folded_twice_vector).all())
        )


if __name__ == "__main__":
    unittest.main()
