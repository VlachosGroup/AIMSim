""" Test the methods of the Molecule class """
from os import remove
import os.path
import unittest

import numpy as np
import rdkit
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolfiles import MolToPDBFile

from molSim.chemical_datastructures import Molecule, MoleculeSet
from molSim.featurize_molecule import Descriptor


class TestMolecule(unittest.TestCase):
    def test_molecule_created_with_no_attributes(self):
        test_molecule = Molecule()
        self.assertIsNone(test_molecule.mol_graph,
                          "Expected attribute mol_graph to be None "
                          "for uninitialized Molecule")
        self.assertIsNone(test_molecule.mol_text,
                          "Expected attribute mol_text to be None "
                          "for uninitialized Molecule")
        self.assertIsNone(test_molecule.mol_property_val,
                          "Expected attribute mol_property_val to be None "
                          "for uninitialized Molecule")
        self.assertIsNone(test_molecule.descriptor.value,
                          "Expected molecule.descriptor.value to be None "
                          "for uninitialized Molecule")

    def test_molecule_created_w_attributes(self):
        test_molecule = Molecule(
                                 mol_text='test_molecule',
                                 mol_property_val=42,
                                 mol_descriptor_val=[1, 2, 3])
        self.assertEqual(test_molecule.mol_text, 'test_molecule',
                         "Expected mol_text attribute to be set.")
        self.assertEqual(test_molecule.mol_property_val, 42,
                         "Expected mol_property_val to be set.")
        self.assertEqual(test_molecule.descriptor.datatype, 'numpy',
                         "Expected descriptor.datatype to be numpy since "
                         "it was initialized by list/array")
        self.assertIsInstance(test_molecule.descriptor.value, np.ndarray,
                              "Expected descriptor.value to be np.ndarray")
        self.assertTrue(np.all(
            test_molecule.descriptor.value == np.array([1, 2, 3])),
                         "Expected descriptor.value to be array[1, 2, 3]")
        self.assertEqual(test_molecule.descriptor.label, 'arbitrary',
                         "Expected descriptor.label to be arbitrary since "
                         "it was initialized by list/array")

    def test_set_molecule_from_smiles(self):
        test_smiles = 'CC'
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        self.assertEqual(test_molecule.mol_text, test_smiles,
                         "Expected mol_text attribute to be set "
                         "to smiles string")
        self.assertIsNotNone(test_molecule.mol_graph,
                             "Expected mol_graph attribute to be set "
                             "from the smiles")
        self.assertIsInstance(test_molecule.mol_graph, rdkit.Chem.rdchem.Mol,
                              "Expected initialized mol_graph to "
                              "be rdkit.Chem.rdchem.Mol object")

    def test_set_molecule_from_file(self):
        test_smiles = 'CC'
        # Case 1: text file
        test_text_molecule = Molecule()
        text_fpath = 'test_mol_src.txt'
        print(f'Creating file {text_fpath}...')
        with open(text_fpath, "w") as fp:
            fp.write(test_smiles+' garbage vals')
        test_text_molecule._set_molecule_from_file(text_fpath)
        self.assertEqual(test_text_molecule.mol_text, test_smiles,
                         "Expected mol_text attribute to be set "
                         "to smiles string when loading from txt file")
        self.assertIsNotNone(test_text_molecule.mol_graph,
                             "Expected mol_graph attribute to be set "
                             "from the smiles when loading from txt file")
        self.assertIsInstance(test_text_molecule.mol_graph,
                              rdkit.Chem.rdchem.Mol,
                              "Expected initialized mol_graph to "
                              "be rdkit.Chem.rdchem.Mol object "
                              "when loading from txt file")
        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)

        # Case 2: pdb file
        test_pdb_molecule = Molecule()
        test_pdb_filename = 'test_mol_src.pdb'
        print(f'Creating file {test_pdb_filename}...')
        test_mol = MolFromSmiles(test_smiles)
        MolToPDBFile(test_mol, test_pdb_filename)
        test_pdb_molecule._set_molecule_from_file(test_pdb_filename)
        self.assertEqual(test_pdb_molecule.mol_text,
                         os.path.basename(test_pdb_filename).split('.')[0],
                         "Expected mol_text attribute to be set "
                         "to smiles string when loading from pdb file")
        self.assertIsNotNone(test_pdb_molecule.mol_graph,
                             "Expected mol_graph attribute to be set "
                             "from the smiles when loading from pdb file")
        self.assertIsInstance(test_pdb_molecule.mol_graph,
                              rdkit.Chem.rdchem.Mol,
                              "Expected initialized mol_graph to "
                              "be rdkit.Chem.rdchem.Mol object "
                              "when loading from pdb file")
        print(f'Test complete. Deleting file {test_pdb_filename}...')
        remove(test_pdb_filename)

    def test_molecule_graph_similar_to_itself_morgan_tanimoto(self):
        test_smiles = 'CC'
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_molecule_duplicate = Molecule()
        test_molecule_duplicate._set_molecule_from_smiles(test_smiles)
        tanimoto_similarity = test_molecule.get_similarity_to_molecule(
                                     test_molecule_duplicate,
                                     similarity_measure='tanimoto',
                                     molecular_descriptor='morgan_fingerprint')
        self.assertEqual(tanimoto_similarity, 1.,
                         "Expected tanimoto similarity to be 1 when comparing "
                         "molecule graph to itself")

    def test_molecule_graph_similar_to_itself_morgan_negl0(self):
        test_smiles = 'CC'
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_molecule_duplicate = Molecule()
        test_molecule_duplicate._set_molecule_from_smiles(test_smiles)
        negl0_similarity = test_molecule.get_similarity_to_molecule(
                                     test_molecule_duplicate,
                                     similarity_measure='neg_l0',
                                     molecular_descriptor='morgan_fingerprint')
        self.assertEqual(negl0_similarity, 0.,
                         "Expected negative L0 norm to be 0 when comparing "
                         "molecule graph to itself")

    def test_molecule_created_with_constructor(self):
        # Molecule created by passing SMILES to constructor
        test_smiles = 'CC'
        test_molecule_from_construct = Molecule(mol_smiles=test_smiles)
        test_molecule_empty = Molecule()
        test_molecule_empty._set_molecule_from_smiles(test_smiles)

    def test_molecule_graph_similar_to_itself_morgan_dice(self):
        test_smiles = 'CC'
        test_molecule = Molecule()
        test_molecule._set_molecule_from_smiles(test_smiles)
        test_molecule_duplicate = Molecule()
        test_molecule_duplicate._set_molecule_from_smiles(test_smiles)
        tanimoto_similarity = test_molecule.get_similarity_to_molecule(
                                     test_molecule_duplicate,
                                     similarity_measure='dice',
                                     molecular_descriptor='morgan_fingerprint')
        self.assertEqual(tanimoto_similarity, 1.,
                         "Expected dice similarity to be 1 when comparing "
                         "molecule graph to itself")


class TestMoleculeSet(unittest.TestCase):
    test_smiles = ['C', 'CC', 'CCC', 'O']
    
    def smiles_seq_to_textfile(self, smiles_seq, property_seq=None):
        text_fpath = 'temp_smiles_seq.txt'
        print(f'Creating text file {text_fpath}')
        with open(text_fpath, "w") as fp:
            for id, smiles in enumerate(smiles_seq):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += ' ' + str(property_seq[id])
                if id < len(smiles_seq) - 1:
                    write_txt += '\n'

                fp.write(write_txt)
        return text_fpath
                
    def test_set_molecule_database_from_textfile(self):
        text_fpath = self.smiles_seq_to_textfile(self.test_smiles)
        molecule_set = MoleculeSet(molecule_database_src=text_fpath,
                                   molecule_database_src_type='text',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from text')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles in text file')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles')
            self.assertIsNone(molecule.mol_property_val,
                              'Expected mol_property_val of Molecule object'
                              'initialized without property to be None')


        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)
    
    def test_set_molecule_database_w_prepoerty_from_textfile(self):
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(self.test_smiles, 
                                                 property_seq=properties)
        molecule_set = MoleculeSet(molecule_database_src=text_fpath,
                                   molecule_database_src_type='text',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from text')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles in text file')
        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)
        

if __name__ == '__main__':
        unittest.main()







