""" Test the methods of the Molecule class """
from os import remove
import os.path
import unittest

import numpy as np
import rdkit
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolfiles import MolToPDBFile

from aimsim.chemical_datastructures import Molecule
from aimsim.exceptions import LoadingError
from aimsim.ops import SimilarityMeasure


class TestMolecule(unittest.TestCase):
    """
    Tests for methods of Molecule class.

    """

    def test_molecule_created_with_no_attributes(self):
        """
        Test for creation of empty Molecule object with no attributes.

        """
        test_molecule = Molecule()
        self.assertIsNone(
            test_molecule.mol_graph,
            "Expected attribute mol_graph to be None for uninitialized Molecule",
        )
        self.assertIsNone(
            test_molecule.mol_text,
            "Expected attribute mol_text to be None for uninitialized Molecule",
        )
        self.assertIsNone(
            test_molecule.mol_property_val,
            "Expected attribute mol_property_val to be None "
            "for uninitialized Molecule",
        )
        self.assertFalse(
            test_molecule.descriptor.check_init(),
            "Expected molecule.descriptor to be unitialized  "
            "for uninitialized Molecule",
        )

    def test_molecule_created_w_attributes(self):
        """
        Test to create Molecule object with descriptor value (list) and a
        response scalar.

        """
        test_molecule = Molecule(
            mol_text="test_molecule", mol_property_val=42, mol_descriptor_val=[1, 2, 3]
        )
        self.assertEqual(
            test_molecule.mol_text,
            "test_molecule",
            "Expected mol_text attribute to be set.",
        )
        self.assertEqual(
            test_molecule.mol_property_val, 42, "Expected mol_property_val to be set."
        )
        self.assertIsInstance(
            test_molecule.descriptor.to_numpy(),
            np.ndarray,
            "Expected descriptor.to_numpy()to be np.ndarray",
        )
        self.assertTrue(
            np.all(test_molecule.descriptor.to_numpy() == np.array([1, 2, 3])),
            "Expected descriptor.to_numpy() to be array[1, 2, 3]",
        )
        self.assertEqual(
            test_molecule.descriptor.label_,
            "arbitrary",
            "Expected descriptor.label to be arbitrary since "
            "it was initialized by list/array",
        )

    def test_set_molecule_from_smiles(self):
        """
        Test to create Molecule object by reading SMILES string.

        """
        test_smiles = "CC"
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        self.assertEqual(
            test_molecule.mol_text,
            test_smiles,
            "Expected mol_text attribute to be set " "to smiles string",
        )
        self.assertIsNotNone(
            test_molecule.mol_graph,
            "Expected mol_graph attribute to be set " "from the smiles",
        )
        self.assertIsInstance(
            test_molecule.mol_graph,
            rdkit.Chem.rdchem.Mol,
            "Expected initialized mol_graph to " "be rdkit.Chem.rdchem.Mol object",
        )

    def test_set_molecule_from_file(self):
        """
        Test to create Molecule object by reading the contents of a file.

        Case #1: text file
        Case #2: PDB file

        """
        test_smiles = "CC"
        # Case 1: text file
        test_text_molecule = Molecule()
        text_fpath = "test_mol_src.txt"
        print(f"Creating file {text_fpath}...")
        with open(text_fpath, "w") as fp:
            fp.write(test_smiles + " garbage vals")
        test_text_molecule._set_molecule_from_file(text_fpath)
        self.assertEqual(
            test_text_molecule.mol_text,
            test_smiles,
            "Expected mol_text attribute to be set "
            "to smiles string when loading from txt file",
        )
        self.assertIsNotNone(
            test_text_molecule.mol_graph,
            "Expected mol_graph attribute to be set "
            "from the smiles when loading from txt file",
        )
        self.assertIsInstance(
            test_text_molecule.mol_graph,
            rdkit.Chem.rdchem.Mol,
            "Expected initialized mol_graph to "
            "be rdkit.Chem.rdchem.Mol object "
            "when loading from txt file",
        )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

        # Case 2: pdb file
        test_pdb_molecule = Molecule()
        test_pdb_filename = "test_mol_src.pdb"
        print(f"Creating file {test_pdb_filename}...")
        test_mol = MolFromSmiles(test_smiles)
        MolToPDBFile(test_mol, test_pdb_filename)
        test_pdb_molecule._set_molecule_from_file(test_pdb_filename)
        self.assertEqual(
            test_pdb_molecule.mol_text,
            os.path.basename(test_pdb_filename).split('.')[0],
            "Expected mol_text attribute to be set "
            "to name of file when loading from pdb file",
        )
        self.assertIsNotNone(
            test_pdb_molecule.mol_graph,
            "Expected mol_graph attribute to be set "
            "from the smiles when loading from pdb file",
        )
        self.assertIsInstance(
            test_pdb_molecule.mol_graph,
            rdkit.Chem.rdchem.Mol,
            "Expected initialized mol_graph to "
            "be rdkit.Chem.rdchem.Mol object "
            "when loading from pdb file",
        )
        print(f"Test complete. Deleting file {test_pdb_filename}...")
        remove(test_pdb_filename)

    def test_molecule_draw(self):
        """
        Test to draw molecule stored in Molecule object.

        """
        test_smiles = "CC"
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_image_fpath = test_smiles + ".png"
        test_molecule.draw(fpath=test_image_fpath)
        self.assertTrue(os.path.isfile(test_image_fpath))
        try:
            print(f"Deleting {test_image_fpath}")
            remove(test_image_fpath)
        except FileNotFoundError:
            print(f"Could not find {test_image_fpath}")

    def test_molecule_graph_similar_to_itself_morgan_tanimoto(self):
        """
        Test that the morgan fingerprint of a Molecule object is similar
        to itself using Tanimoto similarity.

        """
        test_smiles = "CC"
        fingerprint_type = "morgan_fingerprint"
        similarity_metric = "tanimoto"
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_molecule_duplicate = Molecule()
        test_molecule_duplicate._set_molecule_from_smiles(test_smiles)
        test_molecule.set_descriptor(fingerprint_type=fingerprint_type)
        test_molecule_duplicate.set_descriptor(
            fingerprint_type=fingerprint_type)
        similarity_measure = SimilarityMeasure(metric=similarity_metric)
        tanimoto_similarity = test_molecule.get_similarity_to(
            test_molecule_duplicate, similarity_measure=similarity_measure
        )
        self.assertEqual(
            tanimoto_similarity,
            1.0,
            "Expected tanimoto similarity to be 1 when comparing "
            "molecule graph to itself",
        )

    def test_mol_mol_similarity_w_morgan_tanimoto(self):
        """
        Test that the tanimoto similarity of the morgan fingerprints of
        two Molecules are in (0, 1).

        """
        mol1_smiles = "CCCCCCCCC"
        mol2_smiles = "CCCCCCCCCCC"
        fingerprint_type = "morgan_fingerprint"
        similarity_metric = "tanimoto"
        molecules = []
        for smiles in [mol1_smiles, mol2_smiles]:
            molecule = Molecule(mol_smiles=smiles)
            molecule.set_descriptor(fingerprint_type=fingerprint_type)
            molecules.append(molecule)
        similarity_measure = SimilarityMeasure(metric=similarity_metric)
        tanimoto_similarity = molecules[0].get_similarity_to(
            molecules[1], similarity_measure=similarity_measure
        )
        self.assertGreaterEqual(
            tanimoto_similarity, 0.0, "Expected tanimoto similarity to be >= 0."
        )
        self.assertLessEqual(
            tanimoto_similarity, 1.0, "Expected tanimoto similarity to be <= 1."
        )

    def test_molecule_graph_similar_to_itself_morgan_l0(self):
        """
        Test that the morgan fingerprint of a Molecule object is similar
        to itself using L0 norm similarity.

        """
        test_smiles = "CC"
        fingerprint_type = "morgan_fingerprint"
        similarity_metric = "l0_similarity"
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_molecule_duplicate = Molecule()
        test_molecule_duplicate._set_molecule_from_smiles(test_smiles)
        test_molecule.set_descriptor(fingerprint_type=fingerprint_type)
        test_molecule_duplicate.set_descriptor(
            fingerprint_type=fingerprint_type)
        similarity_measure = SimilarityMeasure(metric=similarity_metric)
        negl0_similarity = test_molecule.get_similarity_to(
            test_molecule_duplicate, similarity_measure=similarity_measure
        )
        self.assertEqual(
            negl0_similarity,
            1.0,
            "Expected l0_norm similarity to be 1. when comparing "
            "molecule graph to itself",
        )

    def test_molecule_graph_similar_to_itself_morgan_dice(self):
        """
        Test that the morgan fingerprint of a Molecule object is similar
        to itself using dice similarity.

        """
        test_smiles = "CCO"
        fingerprint_type = "morgan_fingerprint"
        similarity_metric = "dice"
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_molecule_duplicate = Molecule()
        test_molecule_duplicate._set_molecule_from_smiles(test_smiles)
        test_molecule.set_descriptor(fingerprint_type=fingerprint_type)
        test_molecule_duplicate.set_descriptor(
            fingerprint_type=fingerprint_type)
        similarity_measure = SimilarityMeasure(metric=similarity_metric)
        dice_similarity = test_molecule.get_similarity_to(
            test_molecule_duplicate, similarity_measure=similarity_measure
        )
        self.assertEqual(
            dice_similarity,
            1.0,
            "Expected dice similarity to be 1 when comparing "
            "molecule graph to itself",
        )

    def test_mol_src_pdb_loadingerror(self):
        """Error in mol_src should raise LoadingError
        """
        with self.assertRaises(LoadingError):
            Molecule(mol_src='non-existent file.pdb')

    def test_mol_src_txt_loadingerror(self):
        """Error in mol_src should raise LoadingError
        """
        with self.assertRaises(LoadingError):
            Molecule(mol_src='non-existent file.txt')

    def test_mol_smiles_loadingerror(self):
        """Error in mol_smiles should raise LoadingError
        """
        with self.assertRaises(LoadingError):
            Molecule(mol_smiles="XYZ")

    def test_is_same(self):
        """Two identical molecules should be identifed as such.
        """
        mol_1 = Molecule(mol_text="C")
        mol_2 = Molecule(mol_text="C")
        self.assertTrue(Molecule.is_same(mol_1, mol_2))

    def test_get_name(self):
        """Retrieve the name from the molecule.
        """
        mol = Molecule(mol_text="C")
        self.assertEqual(mol.get_name(), "C")

    def test_get_property_value(self):
        """Retrieve the property value from the molecule
        """
        correct_val = 10
        mol = Molecule(mol_text="C", mol_property_val=correct_val)
        self.assertEqual(mol.get_mol_property_val(), correct_val)

    def test_match_fprint_error(self):
        """Trying to match fingerprint from molecule w/o graph should throw an error.
        """
        # initialize a molecule normally
        ref_mol = Molecule(mol_smiles="C")
        ref_mol.set_descriptor(fingerprint_type="morgan_fingerprint")
        # delete the molecular graph
        ref_mol.mol_graph = np.array([])
        mol = Molecule()
        with self.assertRaises(ValueError):
            mol.match_fingerprint_from(ref_mol)


if __name__ == "__main__":
    unittest.main()
