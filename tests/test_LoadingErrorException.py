""" Test the LoadingError Exception """
import unittest

from aimsim.exceptions import LoadingError
from aimsim.chemical_datastructures import Molecule

from pathlib import Path
from os import remove


class TestLoadingERrorException(unittest.TestCase):
    """
    Tests for LoadingError custom exception.

    """

    def test_missing_smiles(self):
        """
        Missing smiles strings should raise a LoadingError.
        """
        with self.assertRaises(LoadingError):
            test_molecule = Molecule()
            test_molecule._set_molecule_from_smiles([])

    def test_invalid_smiles(self):
        """Invalid SMILES strings should raise a LoadingError.
        """
        with self.assertRaises(LoadingError):
            test_molecule = Molecule()
            test_molecule._set_molecule_from_smiles("XYZ")

    def test_missing_pdb(self):
        """Missing PDB files should raise a LoadingError.
        """
        with self.assertRaises(LoadingError):
            test_molecule = Molecule()
            test_molecule._set_molecule_from_pdb("missing.pdb")

    def test_invalid_pdb(self):
        """Invalid PDB files should raise a LoadingError.
        """
        Path('blank.pdb').touch()
        with self.assertRaises(LoadingError):
            test_molecule = Molecule()
            test_molecule._set_molecule_from_pdb("blank.pdb")
        remove('blank.pdb')


if __name__ == "__main__":
    unittest.main()
