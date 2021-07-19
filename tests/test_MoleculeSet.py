"""Test the MoleculeSet class."""
from os import remove, mkdir
import os.path
from shutil import rmtree
import unittest

import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolfiles import MolToPDBFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from molSim.chemical_datastructures import Molecule, MoleculeSet
from molSim.exceptions import NotInitializedError
from molSim.ops import Descriptor, SimilarityMeasure


SUPPORTED_SIMILARITIES = ['tanimoto', 'jaccard', 'negative_l0',
                          'negative_l1', 'negative_l2']
SUPPORTED_FPRINTS = ['morgan_fingerprint', 'topological_fingerprint']


class TestMoleculeSet(unittest.TestCase):
    test_smiles = ['C', 'CC', 'CCC', 'O']
    
    def smiles_seq_to_textfile(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a text file.
        
        Parameters
        ----------
        property_seq : list or np.ndarray
            Optional sequence of molecular responses.
        
        Returns
        -------
        text_fpath : str
            Path to created file.
        
        """
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
        """Helper method to convert a SMILES sequence to a pdb files 
        stored in a directory.
        
        Parameters
        ----------
        property_seq : list or np.ndarray
            Optional sequence of molecular responses.
        
        Returns
        -------
        dir_path : str
            Path to created directory.
        
        """
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
    
    def smiles_seq_to_xl_or_csv(self,
                                ftype,
                                property_seq=None,
                                name_seq=None,
                                feature_arr=None):
        """Helper method to convert a SMILES sequence or arbitrary features 
        to Excel or CSV files.
        
        Parameters
        ----------
        ftype : str
            String label to denote the filetype. 'csv' or 'excel'.
        property_seq : list or np.ndarray
            Optional sequence of molecular responses.
        name_seq : list or np.ndarray
            Optional sequence of molecular names.
        feature_arr : np.ndarray
            Optional array of molecular descriptor values.
        
        Returns
        -------
        fpath : str
            Path to created file.
        
        """
        data = {'feature_smiles': self.test_smiles}
        if property_seq is not None:
            data.update({'response_random': property_seq})
        if name_seq is not None:
            data.update({'feature_name': name_seq})
        if feature_arr is not None:
            feature_arr = np.array(feature_arr)
            for feature_num in range(feature_arr.shape[1]):
                data.update({
                    f'feature_{feature_num}': feature_arr[:, feature_num]})
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
            raise ValueError(f'{ftype} not supported')
        return fpath
                
    def test_set_molecule_database_from_textfile(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        from a textfile.

        """
        text_fpath = self.smiles_seq_to_textfile()
        molecule_set = MoleculeSet(molecule_database_src=text_fpath,
                                   molecule_database_src_type='text',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from text')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles in text file')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(molecule.mol_text, self.test_smiles[id],
                             'Expected mol_text attribute of Molecule object '
                             'to be smiles')
            self.assertIsNone(molecule.mol_property_val,
                              'Expected mol_property_val of Molecule object '
                              'initialized without property to be None')
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)

    def test_subsample_molecule_database_from_textfile(self):
        """
        Test to randomly subsample a molecule database loaded from a textfile.

        """
        text_fpath = self.smiles_seq_to_textfile()
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(molecule_database_src=text_fpath,
                                   molecule_database_src_type='text',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True,
                                   sampling_ratio=sampling_ratio)
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from text')
        self.assertEqual(len(molecule_set.molecule_database),
                         int(sampling_ratio * len(self.test_smiles)),
                         'Expected the size of subsampled database to be equal '
                         'to number of smiles in text file * sampling_ratio')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)
    
    def test_set_molecule_database_w_property_from_textfile(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        and molecular responses from a textfile.
        
        """
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        molecule_set = MoleculeSet(molecule_database_src=text_fpath,
                                   molecule_database_src_type='text',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from text')
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
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {text_fpath}...')
        remove(text_fpath)
    
    def test_set_molecule_database_from_pdb_dir(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        from a directory of pdb files.
        
        """
        dir_path = self.smiles_seq_to_pdb_dir(self.test_smiles)
        molecule_set = MoleculeSet(molecule_database_src=dir_path,
                                   molecule_database_src_type='directory',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from dir')
        self.assertEqual(len(molecule_set.molecule_database), 
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of files in dir')
        for molecule in molecule_set.molecule_database:
            self.assertIn(molecule.mol_text, self.test_smiles,
                          'Expected molecule text to be a smiles string')
            self.assertIsNone(molecule.mol_property_val,
                              'Expected mol_property_val of Molecule object'
                              'initialized without property to be None')
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting directory {dir_path}...')
        rmtree(dir_path)

    def test_subsample_molecule_database_from_pdb_dir(self):
        """
        Test to randomly subsample a molecule database loaded from a 
        directory of pdb files.

        """
        dir_path = self.smiles_seq_to_pdb_dir(self.test_smiles)
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(molecule_database_src=dir_path,
                                   molecule_database_src_type='directory',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True,
                                   sampling_ratio=sampling_ratio)
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from dir')
        self.assertEqual(len(molecule_set.molecule_database),
                         int(sampling_ratio * len(self.test_smiles)),
                         'Expected the size of subsampled database to be '
                         'equal to number of files in dir * sampling_ratio')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting directory {dir_path}...')
        rmtree(dir_path)

    def test_set_molecule_database_from_excel(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        from an Excel file.
        
        """
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype='excel')
        molecule_set = MoleculeSet(molecule_database_src=xl_fpath,
                                   molecule_database_src_type='excel',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
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
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {xl_fpath}...')
        remove(xl_fpath)

    def test_subsample_molecule_database_from_excel(self):
        """
        Test to randomly subsample a molecule database loaded from an 
        Excel file.

        """
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype='excel')
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(molecule_database_src=xl_fpath,
                                   molecule_database_src_type='excel',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True,
                                   sampling_ratio=sampling_ratio)
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
        self.assertEqual(len(molecule_set.molecule_database),
                         int(sampling_ratio * len(self.test_smiles)),
                         'Expected the size of subsampled database to be '
                         'equal to number of smiles * sampling ratio')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {xl_fpath}...')
        remove(xl_fpath)
    
    def test_set_molecule_database_w_property_from_excel(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        and molecular responses from an Excel file.
        
        """
        properties = np.random.normal(size=len(self.test_smiles))
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype='excel', 
                                                property_seq=properties)
        molecule_set = MoleculeSet(molecule_database_src=xl_fpath,
                                   molecule_database_src_type='excel',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
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
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
            print(f'Test complete. Deleting file {xl_fpath}...')
        remove(xl_fpath)

    def test_set_molecule_database_w_descriptor_property_from_excel(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        containing arbitrary molecular descriptor values from an Excel file.
        
        """
        properties = np.random.normal(size=len(self.test_smiles))
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype='excel',
                                                property_seq=properties,
                                                feature_arr=features)
        molecule_set = MoleculeSet(molecule_database_src=xl_fpath,
                                   molecule_database_src_type='excel',
                                   similarity_measure='negative_l0',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose,
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
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
            self.assertTrue((molecule.descriptor.to_numpy()
                             == features[id]).all,
                            'Expected descriptor value to be same as the '
                            'vector used to initialize descriptor')
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
            print(f'Test complete. Deleting file {xl_fpath}...')
        remove(xl_fpath)

    def test_set_molecule_database_from_csv(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        and molecular responses from a CSV file.
        
        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv')
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'csv file')
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
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)

    def test_subsample_molecule_database_from_csv(self):
        """
        Test to randomly subsample a molecule database loaded from an 
        CSV file.

        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv')
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   sampling_ratio=sampling_ratio,
                                   is_verbose=True)
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'csv file')
        self.assertEqual(len(molecule_set.molecule_database),
                         int(sampling_ratio * len(self.test_smiles)),
                         'Expected the size of database to be equal to number '
                         'of smiles * sampling_ratio')
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_set_molecule_database_w_property_from_csv(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        and molecular responses from a CSV file.
        
        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                 property_seq=properties)
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   fingerprint_type='morgan_fingerprint',
                                   similarity_measure='tanimoto',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose, 
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'csv file')
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
            self.assertIsInstance(molecule, Molecule)
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)

    def test_set_molecule_database_w_descriptor_property_from_csv(self):
        """
        Test to create MoleculeSet object by reading molecule database 
        containing arbitrary molecular descriptors and molecular responses 
        from a CSV file.
        
        """
        properties = np.random.normal(size=len(self.test_smiles))
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv',
                                                 property_seq=properties,
                                                 feature_arr=features)
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   similarity_measure='negative_l0',
                                   is_verbose=True)
        self.assertTrue(molecule_set.is_verbose,
                        'Expected is_verbose to be True')
        self.assertIsNotNone(molecule_set.molecule_database,
                             'Expected molecule_database to be set from '
                             'excel file')
        self.assertEqual(len(molecule_set.molecule_database),
                         len(self.test_smiles),
                         'Expected the size of database to be equal to number '
                         'of smiles in csv file')
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
            self.assertTrue((molecule.descriptor.to_numpy()
                             == features[id]).all,
                            'Expected descriptor value to be same as the '
                            'vector used to initialize descriptor')
            self.assertIsInstance(molecule, Molecule,
                                  'Expected member of molecule_set to '
                                  'be Molecule object')
            print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)

    def test_set_molecule_database_w_similarity_from_csv(self):
        """
        Verify that a NotInitializedError is raised if no fingerprint_type 
        is specified when instantiating a MoleculeSet object.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                 property_seq=properties)
        for similarity_measure in SUPPORTED_SIMILARITIES:
            with self.assertRaises(NotInitializedError):
                MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type='csv',
                    similarity_measure=similarity_measure,
                    is_verbose=True)

        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_set_molecule_database_fingerprint_from_csv(self):
        """
        Verify that a TypeError is raised if no similarity_measure
        is specified when instantiating a MoleculeSet object.
        
        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                 property_seq=properties)
        for descriptor in SUPPORTED_FPRINTS:
            with self.assertRaises(TypeError):
                MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type='csv',
                    fingerprint_type=descriptor,
                    is_verbose=True)

        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_set_molecule_database_w_fingerprint_similarity_from_csv(self):
        """
        Test all combinations of fingerprints and similarity measures with the
        MoleculeSet class.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv', 
                                                 property_seq=properties)
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                                        molecule_database_src=csv_fpath,
                                        molecule_database_src_type='csv',
                                        fingerprint_type=descriptor,
                                        similarity_measure=similarity_measure,
                                        is_verbose=True)
                self.assertTrue(molecule_set.is_verbose, 
                                'Expected is_verbose to be True')
                self.assertIsNotNone(molecule_set.molecule_database,
                                     'Expected molecule_database to '
                                     'be set from csv file')
                for molecule in molecule_set.molecule_database:
                    self.assertTrue(molecule.descriptor.check_init(),
                                    'Expected descriptor to be set')
                self.assertIsNotNone(molecule_set.similarity_matrix,
                                     'Expected similarity_matrix to be set')
        print(f'Test complete. Deleting file {csv_fpath}...')
        remove(csv_fpath)
    
    def test_get_most_similar_pairs(self):
        """
        Test that all combinations of fingerprint_type and similarity measure
        works with the MoleculeSet.get_most_similar_pairs() method.

        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv')
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                                       molecule_database_src=csv_fpath,
                                       molecule_database_src_type='csv',
                                       fingerprint_type=descriptor,
                                       similarity_measure=similarity_measure,
                                       is_verbose=True)
                molecule_pairs = molecule_set.get_most_similar_pairs()
                self.assertIsInstance(molecule_pairs, list, 
                                      'Expected get_most_similar_pairs() '
                                      'to return list')
                for pair in molecule_pairs:
                    self.assertIsInstance(pair, tuple, 
                                          'Expected elements of list '
                                          'returned by get_most_similar_pairs()'
                                          ' to be tuples')
    
    def test_get_most_dissimilar_pairs(self):
        """
        Test that all combinations of fingerprint_type and similarity measure
        works with the MoleculeSet.get_most_dissimilar_pairs() method.
        
        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv')
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                                        molecule_database_src=csv_fpath,
                                        molecule_database_src_type='csv',
                                        fingerprint_type=descriptor,
                                        similarity_measure=similarity_measure,
                                        is_verbose=True)
                molecule_pairs = molecule_set.get_most_dissimilar_pairs()
                self.assertIsInstance(molecule_pairs, list, 
                                      'Expected get_most_dissimilar_pairs() '
                                      'to return list')
                for pair in molecule_pairs:
                    self.assertIsInstance(pair, tuple, 
                                          'Expected elements of list returned'
                                          ' by get_most_dissimilar_pairs() '
                                          'to be tuples')
    
    def test_pca_transform(self):
        """ 
        Test the unsupervised transformation of molecules in 
        MoleculSet using Principal Component Analysis.
        
        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype='csv',
                                                 feature_arr=features)
        molecule_set = MoleculeSet(molecule_database_src=csv_fpath,
                                   molecule_database_src_type='csv',
                                   similarity_measure='negative_l0',
                                   is_verbose=True)
        features = StandardScaler().fit_transform(features)
        features = PCA().fit_transform(features)
        error_matrix = features - molecule_set.get_transformed_descriptors()
        error_threshold = 1e-6
        self.assertLessEqual(error_matrix.min(), error_threshold,
                            'Expected transformed molecular descriptors to be '
                            'equal to PCA decomposed features')
        

if __name__ == '__main__':
    unittest.main()
