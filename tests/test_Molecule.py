""" Test the methods of the Molecule class """
from os import remove, mkdir
import os.path
from  shutil import rmtree
import unittest

import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolfiles import MolToPDBFile

from molSim.chemical_datastructures import Molecule, MoleculeSet
from molSim.featurize_molecule import Descriptor
from molSim.similarity_measures import get_supported_measures


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
        test_molecule = Molecule(mol_text='test_molecule',
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
    
    def smiles_seq_to_textfile(self, property_seq=None):
        text_fpath = 'temp_smiles_seq.txt'
        print(f'Creating text file {text_fpath}')
        with open(text_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += ' ' + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += '\n'

                fp.write(write_txt)
        return text_fpath
    
    def smiles_seq_to_pdb_dir(self, property_seq=None):
        dir_path = 'test_dir'
        if not os.path.isdir(dir_path):
            print(f'Creating directory {dir_path}')
            mkdir(dir_path)
        for smiles_str in self.test_smiles:
            mol_graph = MolFromSmiles(smiles_str)
            assert mol_graph is not None
            pdb_fpath = os.path.join(dir_path, smiles_str + '.pdb')
            print(f'Creating file {pdb_fpath}')
            MolToPDBFile(mol_graph, pdb_fpath)
        return dir_path
    
    def smiles_seq_to_xl_or_csv(self, ftype, property_seq=None, name_seq=None):        
        data = {'feature_smiles': self.test_smiles}
        if property_seq is not None:
            data.update({'response_random': property_seq})
        if name_seq is not None:
            data.update({'feature_name': name_seq})
        data_df = pd.DataFrame(data)
        fpath = 'temp_mol_file'        
        if ftype == 'excel':
            fpath += '.xlsx'
            print(f'Creating {ftype} file {fpath}')
            data_df.to_excel(fpath)
        elif ftype == 'csv':
            fpath += '.csv'
            print(f'Creating {ftype} file {fpath}')
            data_df.to_csv(fpath)
        else:
            raise NotImplementedError(f'{ftype} not supported')  
        return fpath
                
    def test_set_molecule_database_from_textfile(self):
        text_fpath = self.smiles_seq_to_textfile()
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
    
    def test_set_molecule_database_w_property_from_textfile(self):
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
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
            self.assertAlmostEqual(molecule.mol_property_val, 
                                   properties[id],
                                   places=7,
                                   msg='Expected mol_property_val of' 
                                        'Molecule object '
                                        'to be set to value in text file')
        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)
    
    def test_set_molecule_database_from_pdb_dir(self):
        dir_path = self.smiles_seq_to_pdb_dir(self.test_smiles)
        molecule_set = MoleculeSet(molecule_database_src=dir_path,
                                   molecule_database_src_type='directory',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from dir')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of files in dir')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles')
            self.assertIsNone(molecule.mol_property_val,
                              'Expected mol_property_val of Molecule object'
                              'initialized without property to be None')
        print(f'Test complete. Deleting directory {dir_path}...')
        rmtree(dir_path)

    def test_set_molecule_database_from_excel(self):
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype='excel')
        molecule_set = MoleculeSet(molecule_database_src=xl_fpath,
                                   molecule_database_src_type='excel',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles when names not present in excel')
            self.assertIsNone(molecule.mol_property_val,
                              'Expected mol_property_val of Molecule object'
                              'initialized without property to be None')
        print(f'Test complete. Deleting file {xl_fpath}...')
        remove(xl_fpath)
    
    def test_set_molecule_database_w_property_from_excel(self):
        properties = np.random.normal(size=len(self.test_smiles))
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype='excel', 
                                                property_seq=properties)
        molecule_set = MoleculeSet(molecule_database_src=xl_fpath,
                                   molecule_database_src_type='excel',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles in excel file')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles when names not present in excel')
            self.assertAlmostEqual(molecule.mol_property_val, 
                                   properties[id],
                                   places=7,
                                   msg='Expected mol_property_val of' 
                                        'Molecule object '
                                        'to be set to value in excel file')
        print(f'Test complete. Deleting file {xl_fpath}...')
        remove(xl_fpath)

    def test_set_molecule_database_from_csv(self):
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv')
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'csv file')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles when names not present in csv')
            self.assertIsNone(molecule.mol_property_val,
                              'Expected mol_property_val of Molecule object'
                              'initialized without property to be None')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_set_molecule_database_w_property_from_csv(self):
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                property_seq=properties)
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'csvfile')
        self.assertIsNone(molecule_set.molecular_descriptor,
                          'Expected molecular_descriptor to be unset')
        self.assertIsNone(molecule_set.similarity_measure,
                          'Expected similarity_measure to be unset')
        self.assertIsNone(molecule_set.similarity_matrix,
                          'Expected similarity_matrix to be unset')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles when names not present in csv')
            self.assertAlmostEqual(molecule.mol_property_val, 
                                   properties[id],
                                   places=7,
                                   msg='Expected mol_property_val of' 
                                        'Molecule object '
                                        'to be set to value in csv file')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)

    def test_set_molecule_database_w_similarity_from_csv(self):
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                property_seq=properties)
        for similarity_measure in get_supported_measures():
            molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                    molecule_database_src_type='csv',
                                    similarity_measure=similarity_measure,
                                    is_verbose=True)
            self.assertTrue(molecule_set.is_verbose, 
                            'Expected is_verbose to be True')
            self.assertIsNotNone(molecule_set.molecule_database,
                                'Expected molecule_database to be set from '
                                'csvfile')
            self.assertIsNone(molecule_set.molecular_descriptor,
                            'Expected molecular_descriptor to be unset')
            self.assertEqual(molecule_set.similarity_measure, 
                             similarity_measure,
                            'Expected similarity measure attribute of '
                            'molecule_set to be the same as the initial value')
            self.assertIsNone(molecule_set.similarity_matrix,
                            'Expected similarity_matrix to be unset')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_set_molecule_database_descriptor_from_csv(self):
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                property_seq=properties)
        for descriptor in Descriptor().get_supported_descriptors():
            molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                    molecule_database_src_type='csv',
                                    molecular_descriptor=descriptor,
                                    is_verbose=True)
            self.assertTrue(molecule_set.is_verbose, 
                            'Expected is_verbose to be True')
            self.assertIsNotNone(molecule_set.molecule_database,
                                'Expected molecule_database to be set from '
                                'csvfile')
            self.assertIsNone(molecule_set.similarity_measure,
                             'Expected similarity_measure to be unset')
            self.assertEqual(molecule_set.molecular_descriptor, 
                             descriptor,
                            'Expected molecular_descriptor attribute of '
                            'molecule_set to be the same as the initial value')
            self.assertIsNone(molecule_set.similarity_matrix,
                            'Expected similarity_matrix to be unset')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_set_molecule_database_w_descriptor_similarity_from_csv(self):
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                property_seq=properties)
        for descriptor in Descriptor().get_supported_descriptors():
            for similarity_measure in get_supported_measures():
                molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                        molecule_database_src_type='csv',
                                        molecular_descriptor=descriptor,
                                        similarity_measure=similarity_measure,
                                        is_verbose=True)
                self.assertTrue(molecule_set.is_verbose, 
                                'Expected is_verbose to be True')
                self.assertIsNotNone(molecule_set.molecule_database,
                                    'Expected molecule_database to be set from '
                                    'csvfile')
                self.assertIsNotNone(molecule_set.similarity_measure,
                                     'Expected similarity_measure to be set')
                self.assertIsNotNone(molecule_set.similarity_matrix,
                                     'Expected similarity_matrix to be set')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
        

if __name__ == '__main__':
        unittest.main()







