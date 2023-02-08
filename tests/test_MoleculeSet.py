"""Test the MoleculeSet class."""
from os import remove, mkdir
import os.path
from shutil import rmtree
import unittest

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolfiles import MolToPDBFile
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler

from aimsim.chemical_datastructures import Molecule, MoleculeSet
from aimsim.ops import Descriptor, SimilarityMeasure
from aimsim.exceptions import NotInitializedError, InvalidConfigurationError


SUPPORTED_SIMILARITIES = SimilarityMeasure.get_supported_metrics()

SUPPORTED_FPRINTS = Descriptor.get_supported_fprints()


class TestMoleculeSet(unittest.TestCase):
    test_smarts = [
        "[CH3:1][S:2][c:3]1[cH:4][cH:5][c:6]([B:7]([OH:8])[OH:9])[cH:10][cH:11]1",
        "[NH:1]1[CH2:2][CH2:3][O:4][CH2:5][CH2:6]1.[O:7]=[S:8]=[O:9]",
    ]

    test_smiles = [
        "CCCCCCC",
        "CCCC",
        "CCC",
        "CO",
        "CN",
        "C1=CC=CC=C1",
        "CC1=CC=CC=C1",
        "C(=O)(N)N",
    ]

    def get_feature_set(self, dimensionality=10):
        n_test_mols = len(self.test_smiles)
        test_feature_set = []
        feature_value_upper_limit = 1000
        for i in range(n_test_mols):
            test_feature_set.append(
                [
                    np.random.random() * feature_value_upper_limit
                    for _ in range(dimensionality)
                ]
            )
        return np.array(test_feature_set)

    def smiles_seq_to_textfile(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a text file.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses.. Defaults to None.

        Returns:
            str: Path to created file.
        """
        text_fpath = "temp_smiles_seq.txt"
        print(f"Creating text file {text_fpath}")
        with open(text_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)
        return text_fpath

    def smiles_seq_to_smi_file(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a .smi file.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses. Defaults to None.

        Returns:
            str: Path to created file.
        """
        smi_fpath = "temp_smiles_seq.smi"
        print(f"Creating text file {smi_fpath}")
        with open(smi_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)
        return smi_fpath

    def smiles_seq_to_smiles_file(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a .SMILES file.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses. Defaults to None.

        Returns:
            str: Path to created file.
        """
        SMILES_fpath = "temp_smiles_seq.SMILES"
        print(f"Creating text file {SMILES_fpath}")
        with open(SMILES_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smiles):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)
        return SMILES_fpath

    def smarts_seq_to_smiles_file(self, property_seq=None):
        """Helper method to convert a SMARTS sequence to a .SMILES file.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses. Defaults to None.

        Returns:
            str: Path to created file.
        """
        SMILES_fpath = "temp_smiles_seq.SMILES"
        print(f"Creating text file {SMILES_fpath}")
        with open(SMILES_fpath, "w") as fp:
            for id, smiles in enumerate(self.test_smarts):
                write_txt = smiles
                if property_seq is not None:
                    write_txt += " " + str(property_seq[id])
                if id < len(self.test_smiles) - 1:
                    write_txt += "\n"

                fp.write(write_txt)
        return SMILES_fpath

    def smiles_seq_to_pdb_dir(self, property_seq=None):
        """Helper method to convert a SMILES sequence to a pdb files
        stored in a directory.

        Args:
            property_seq (list or np.ndarray, optional): Optional sequence of
                molecular responses. Defaults to None.

        Returns:
            str: Path to created directory.
        """
        dir_path = "test_dir"
        if not os.path.isdir(dir_path):
            print(f"Creating directory {dir_path}")
            mkdir(dir_path)
        for smiles_str in self.test_smiles:
            mol_graph = MolFromSmiles(smiles_str)
            assert mol_graph is not None
            pdb_fpath = os.path.join(dir_path, smiles_str + ".pdb")
            print(f"Creating file {pdb_fpath}")
            MolToPDBFile(mol_graph, pdb_fpath)
        return dir_path

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
        data = {"descriptor_smiles": self.test_smiles}
        if property_seq is not None:
            data.update({"response_random": property_seq})
        if name_seq is not None:
            data.update({"descriptor_name": name_seq})
        if feature_arr is not None:
            feature_arr = np.array(feature_arr)
            for feature_num in range(feature_arr.shape[1]):
                data.update({f"descriptor_{feature_num}": feature_arr[:, feature_num]})
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

    def test_set_molecule_database_from_textfile(self):
        """
        Test to create MoleculeSet object by reading molecule database
        from a textfile.

        """
        text_fpath = self.smiles_seq_to_textfile()
        molecule_set = MoleculeSet(
            molecule_database_src=text_fpath,
            molecule_database_src_type="text",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from text",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in text file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object to be smiles",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object "
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

    def test_set_molecule_database_from_smi_file(self):
        """
        Test to create MoleculeSet object by reading molecule database
        from a smi file.

        """
        text_fpath = self.smiles_seq_to_smi_file()
        molecule_set = MoleculeSet(
            molecule_database_src=text_fpath,
            molecule_database_src_type="text",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from text",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in text file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object to be smiles",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object "
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

    def test_set_molecule_database_from_smiles_file(self):
        """
        Test to create MoleculeSet object by reading molecule database
        from a SMILES file.

        """
        text_fpath = self.smiles_seq_to_smiles_file()
        molecule_set = MoleculeSet(
            molecule_database_src=text_fpath,
            molecule_database_src_type="text",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from text",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in text file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object to be smiles",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object "
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

    def test_set_molecule_database_from_smarts_file(self):
        """
        Test to create MoleculeSet object by reading molecule database
        from a SMILES file containing SMARTS strings.

        """
        text_fpath = self.smarts_seq_to_smiles_file()
        molecule_set = MoleculeSet(
            molecule_database_src=text_fpath,
            molecule_database_src_type="text",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from text",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smarts),
            "Expected the size of database to be equal to number "
            "of smiles in text file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smarts[id],
                "Expected mol_text attribute of Molecule object to be smiles",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object "
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

    def test_subsample_molecule_database_from_textfile(self):
        """
        Test to randomly subsample a molecule database loaded from a textfile.

        """
        text_fpath = self.smiles_seq_to_textfile()
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(
            molecule_database_src=text_fpath,
            molecule_database_src_type="text",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
            sampling_ratio=sampling_ratio,
        )
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from text",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            int(sampling_ratio * len(self.test_smiles)),
            "Expected the size of subsampled database to be equal "
            "to number of smiles in text file * sampling_ratio",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

    def test_set_molecule_database_w_property_from_textfile(self):
        """
        Test to create MoleculeSet object by reading molecule database
        and molecular responses from a textfile.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        text_fpath = self.smiles_seq_to_textfile(property_seq=properties)
        molecule_set = MoleculeSet(
            molecule_database_src=text_fpath,
            molecule_database_src_type="text",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from text",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in text file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object " "to be smiles",
            )
            self.assertAlmostEqual(
                molecule.mol_property_val,
                properties[id],
                places=7,
                msg="Expected mol_property_val of"
                "Molecule object "
                "to be set to value in text file",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting file {text_fpath}...")
        remove(text_fpath)

    def test_set_molecule_database_from_pdb_dir(self):
        """
        Test to create MoleculeSet object by reading molecule database
        from a directory of pdb files.

        """
        dir_path = self.smiles_seq_to_pdb_dir(self.test_smiles)
        molecule_set = MoleculeSet(
            molecule_database_src=dir_path,
            molecule_database_src_type="directory",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from dir",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number " "of files in dir",
        )
        for molecule in molecule_set.molecule_database:
            self.assertIn(
                molecule.mol_text,
                self.test_smiles,
                "Expected molecule text to be a smiles string",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object"
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting directory {dir_path}...")
        rmtree(dir_path)

    def test_subsample_molecule_database_from_pdb_dir(self):
        """
        Test to randomly subsample a molecule database loaded from a
        directory of pdb files.

        """
        dir_path = self.smiles_seq_to_pdb_dir(self.test_smiles)
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(
            molecule_database_src=dir_path,
            molecule_database_src_type="directory",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
            sampling_ratio=sampling_ratio,
        )
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from dir",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            int(sampling_ratio * len(self.test_smiles)),
            "Expected the size of subsampled database to be "
            "equal to number of files in dir * sampling_ratio",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
        print(f"Test complete. Deleting directory {dir_path}...")
        rmtree(dir_path)

    def test_set_molecule_database_from_excel(self):
        """
        Test to create MoleculeSet object by reading molecule database
        from an Excel file.

        """
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype="excel")
        molecule_set = MoleculeSet(
            molecule_database_src=xl_fpath,
            molecule_database_src_type="excel",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from excel file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number of smiles",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object "
                "to be smiles when names not present in excel",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object"
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to be Molecule object",
            )
        print(f"Test complete. Deleting file {xl_fpath}...")
        remove(xl_fpath)

    def test_set_molecule_database_from_excel_without_smiles_name(self):
        xl_fpath = "test_files.xlsx"
        features = self.get_feature_set()  # n_samples x dimensionality
        data = dict()
        for feature_id in range(features.shape[-1]):
            data[f"descriptor_{feature_id}"] = features[:, feature_id].flatten()
        df = pd.DataFrame(data)
        print(f"Creating text file {xl_fpath}")
        df.to_excel(xl_fpath)
        molecule_set = MoleculeSet(
            molecule_database_src=xl_fpath,
            molecule_database_src_type="excel",
            similarity_measure="l2_similarity",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from excel file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number of smiles",
        )
        self.assertTrue(
            np.allclose(features, molecule_set.get_mol_features(), rtol=1e-4),
            "Expected molecule feature vectors to "
            "be the same as features in input file",
        )
        with self.assertRaises(ValueError):
            molecule_set = MoleculeSet(
                molecule_database_src=xl_fpath,
                molecule_database_src_type="excel",
                fingerprint_type="morgan_fingerprint",
                similarity_measure="l2_similarity",
                is_verbose=True,
            )
        with self.assertRaises(ValueError):
            molecule_set = MoleculeSet(
                molecule_database_src=xl_fpath,
                molecule_database_src_type="excel",
                similarity_measure="tanimoto_similarity",
                is_verbose=True,
            )
        print(f"Test complete. Deleting file {xl_fpath}...")
        remove(xl_fpath)

    def test_subsample_molecule_database_from_excel(self):
        """
        Test to randomly subsample a molecule database loaded from an
        Excel file.

        """
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype="excel")
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(
            molecule_database_src=xl_fpath,
            molecule_database_src_type="excel",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
            sampling_ratio=sampling_ratio,
        )
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from excel file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            int(sampling_ratio * len(self.test_smiles)),
            "Expected the size of subsampled database to be "
            "equal to number of smiles * sampling ratio",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to be Molecule object",
            )
        print(f"Test complete. Deleting file {xl_fpath}...")
        remove(xl_fpath)

    def test_set_molecule_database_w_property_from_excel(self):
        """
        Test to create MoleculeSet object by reading molecule database
        and molecular responses from an Excel file.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        xl_fpath = self.smiles_seq_to_xl_or_csv(ftype="excel", property_seq=properties)
        molecule_set = MoleculeSet(
            molecule_database_src=xl_fpath,
            molecule_database_src_type="excel",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from excel file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in excel file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object "
                "to be smiles when names not present in excel",
            )
            self.assertAlmostEqual(
                molecule.mol_property_val,
                properties[id],
                places=7,
                msg="Expected mol_property_val of"
                "Molecule object "
                "to be set to value in excel file",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to be Molecule object",
            )
            print(f"Test complete. Deleting file {xl_fpath}...")
        remove(xl_fpath)

    def test_set_molecule_database_w_descriptor_property_from_excel(self):
        """
        Test to create MoleculeSet object by reading molecule database
        containing arbitrary molecular descriptor values from an Excel file.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        xl_fpath = self.smiles_seq_to_xl_or_csv(
            ftype="excel", property_seq=properties, feature_arr=features
        )
        molecule_set = MoleculeSet(
            molecule_database_src=xl_fpath,
            molecule_database_src_type="excel",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from " "excel file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in excel file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object "
                "to be smiles when names not present in excel",
            )
            self.assertAlmostEqual(
                molecule.mol_property_val,
                properties[id],
                places=7,
                msg="Expected mol_property_val of"
                "Molecule object "
                "to be set to value in excel file",
            )
            self.assertTrue(
                (molecule.descriptor.to_numpy() == features[id]).all,
                "Expected descriptor value to be same as the "
                "vector used to initialize descriptor",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to " "be Molecule object",
            )
            print(f"Test complete. Deleting file {xl_fpath}...")
        remove(xl_fpath)

    def test_set_molecule_database_from_csv(self):
        """
        Test to create MoleculeSet object by reading molecule database
        and molecular responses from a CSV file.

        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv")
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from " "csv file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number " "of smiles",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object "
                "to be smiles when names not present in csv",
            )
            self.assertIsNone(
                molecule.mol_property_val,
                "Expected mol_property_val of Molecule object"
                "initialized without property to be None",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to be Molecule object",
            )
        print(f"Test complete. Deleting file {csv_fpath}...")
        remove(csv_fpath)

    def test_subsample_molecule_database_from_csv(self):
        """
        Test to randomly subsample a molecule database loaded from an
        CSV file.

        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv")
        sampling_ratio = 0.5
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            sampling_ratio=sampling_ratio,
            is_verbose=True,
        )
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from csv file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            int(sampling_ratio * len(self.test_smiles)),
            "Expected the size of database to be equal to number "
            "of smiles * sampling_ratio",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to be Molecule object",
            )
        print(f"Test complete. Deleting file {csv_fpath}...")
        remove(csv_fpath)

    def test_set_molecule_database_w_property_from_csv(self):
        """
        Test to create MoleculeSet object by reading molecule database
        and molecular responses from a CSV file.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", property_seq=properties)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from csv file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object "
                "to be smiles when names not present in csv",
            )
            self.assertAlmostEqual(
                molecule.mol_property_val,
                properties[id],
                places=7,
                msg="Expected mol_property_val of"
                "Molecule object "
                "to be set to value in csv file",
            )
            self.assertIsInstance(molecule, Molecule)
        print(f"Test complete. Deleting file {csv_fpath}...")
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
        csv_fpath = self.smiles_seq_to_xl_or_csv(
            ftype="csv", property_seq=properties, feature_arr=features
        )
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        self.assertTrue(molecule_set.is_verbose, "Expected is_verbose to be True")
        self.assertIsNotNone(
            molecule_set.molecule_database,
            "Expected molecule_database to be set from " "excel file",
        )
        self.assertEqual(
            len(molecule_set.molecule_database),
            len(self.test_smiles),
            "Expected the size of database to be equal to number "
            "of smiles in csv file",
        )
        for id, molecule in enumerate(molecule_set.molecule_database):
            self.assertEqual(
                molecule.mol_text,
                self.test_smiles[id],
                "Expected mol_text attribute of Molecule object "
                "to be smiles when names not present in csv",
            )
            self.assertAlmostEqual(
                molecule.mol_property_val,
                properties[id],
                places=7,
                msg="Expected mol_property_val of"
                "Molecule object "
                "to be set to value in csv file",
            )
            self.assertTrue(
                (molecule.descriptor.to_numpy() == features[id]).all,
                "Expected descriptor value to be same as the "
                "vector used to initialize descriptor",
            )
            self.assertIsInstance(
                molecule,
                Molecule,
                "Expected member of molecule_set to be Molecule object",
            )
        print(f"Test complete. Deleting file {csv_fpath}...")
        remove(csv_fpath)

    def test_set_molecule_database_w_similarity_from_csv(self):
        """
        Verify that a NotInitializedError is raised if no fingerprint_type
        is specified when instantiating a MoleculeSet object.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", property_seq=properties)
        for similarity_measure in SUPPORTED_SIMILARITIES:
            with self.assertRaises(NotInitializedError):
                MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type="csv",
                    similarity_measure=similarity_measure,
                    is_verbose=False,
                )

        print(f"Test complete. Deleting file {csv_fpath}...")
        remove(csv_fpath)

    def test_set_molecule_database_fingerprint_from_csv(self):
        """
        Verify that a TypeError is raised if no similarity_measure
        is specified when instantiating a MoleculeSet object.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", property_seq=properties)
        for descriptor in SUPPORTED_FPRINTS:
            with self.assertRaises(TypeError):
                MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type="csv",
                    fingerprint_type=descriptor,
                    is_verbose=False,
                )

        print(f"Test complete. Deleting file {csv_fpath}...")
        remove(csv_fpath)

    def test_set_molecule_database_w_fingerprint_similarity_from_csv(self):
        """
        Test all combinations of fingerprints and similarity measures with the
        MoleculeSet class.

        """
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", property_seq=properties)
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type="csv",
                    fingerprint_type=descriptor,
                    similarity_measure=similarity_measure,
                    is_verbose=False,
                )
                self.assertFalse(
                    molecule_set.is_verbose, "Expected is_verbose to be False"
                )
                self.assertIsNotNone(
                    molecule_set.molecule_database,
                    "Expected molecule_database to be set from csv file",
                )
                for molecule in molecule_set.molecule_database:
                    self.assertTrue(
                        molecule.descriptor.check_init(),
                        "Expected descriptor to be set",
                    )
                self.assertIsNotNone(
                    molecule_set.similarity_matrix,
                    "Expected similarity_matrix to be set",
                )
        print(f"Test complete. Deleting file {csv_fpath}...")
        remove(csv_fpath)

    def test_get_most_similar_pairs(self):
        """
        Test that all combinations of fingerprint_type and similarity measure
        works with the MoleculeSet.get_most_similar_pairs() method.

        """
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
                molecule_pairs = molecule_set.get_most_similar_pairs()
                self.assertIsInstance(
                    molecule_pairs,
                    list,
                    "Expected get_most_similar_pairs() to return list",
                )
                for pair in molecule_pairs:
                    self.assertIsInstance(
                        pair,
                        tuple,
                        "Expected elements of list "
                        "returned by get_most_similar_pairs()"
                        " to be tuples",
                    )
        remove(csv_fpath)

    def test_get_most_dissimilar_pairs(self):
        """
        Test that all combinations of fingerprint_type and similarity measure
        works with the MoleculeSet.get_most_dissimilar_pairs() method.

        """
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
                molecule_pairs = molecule_set.get_most_dissimilar_pairs()
                self.assertIsInstance(
                    molecule_pairs,
                    list,
                    "Expected get_most_dissimilar_pairs() " "to return list",
                )
                for pair in molecule_pairs:
                    self.assertIsInstance(
                        pair,
                        tuple,
                        "Expected elements of list returned"
                        " by get_most_dissimilar_pairs() "
                        "to be tuples",
                    )
        remove(csv_fpath)

    def test_pca_transform(self):
        """
        Test the unsupervised transformation of molecules in
        MoleculSet using Principal Component Analysis.

        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", feature_arr=features)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        features = StandardScaler().fit_transform(features)
        features = PCA(n_components=2).fit_transform(features)
        error_matrix = features - molecule_set.get_transformed_descriptors()
        error_threshold = 1e-6
        self.assertLessEqual(
            error_matrix.min(),
            error_threshold,
            "Expected transformed molecular descriptors to be "
            "equal to PCA decomposed features",
        )
        remove(csv_fpath)

    def test_mds_transform(self):
        """
        Test the unsupervised transformation of molecules in
        MoleculSet using MDS.

        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", feature_arr=features)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        features = StandardScaler().fit_transform(features)
        features = MDS().fit_transform(features)
        error_matrix = features - molecule_set.get_transformed_descriptors(
            method_="mds"
        )
        error_threshold = 1e-6
        self.assertLessEqual(
            error_matrix.min(),
            error_threshold,
            "Expected transformed molecular descriptors to be "
            "equal to MDS decomposed features",
        )
        remove(csv_fpath)

    def test_tsne_transform(self):
        """
        Test the unsupervised transformation of molecules in
        MoleculSet using TSNE.

        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", feature_arr=features)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        features = StandardScaler().fit_transform(features)
        features = TSNE(perplexity=len(self.test_smiles) / 2).fit_transform(features)
        error_matrix = features - molecule_set.get_transformed_descriptors(
            method_="tsne",
            perplexity=1,
        )
        error_threshold = 1e-6
        self.assertLessEqual(
            error_matrix.min(),
            error_threshold,
            "Expected transformed molecular descriptors to be "
            "equal to TSNE decomposed features",
        )
        remove(csv_fpath)

    def test_isomap_transform(self):
        """
        Test the unsupervised transformation of molecules in
        MoleculSet using Isomap.

        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", feature_arr=features)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        features = StandardScaler().fit_transform(features)
        features = Isomap().fit_transform(features)
        error_matrix = features - molecule_set.get_transformed_descriptors(
            method_="isomap"
        )
        error_threshold = 1e-6
        self.assertLessEqual(
            error_matrix.min(),
            error_threshold,
            "Expected transformed molecular descriptors to be "
            "equal to Isomap decomposed features",
        )
        remove(csv_fpath)

    def test_spectral_embedding_transform(self):
        """
        Test the unsupervised transformation of molecules in
        MoleculSet using Isomap.

        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", feature_arr=features)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        features = StandardScaler().fit_transform(features)
        features = SpectralEmbedding().fit_transform(features)
        error_matrix = features - molecule_set.get_transformed_descriptors(
            method_="spectral_embedding"
        )
        error_threshold = 1e-6
        self.assertLessEqual(
            error_matrix.min(),
            error_threshold,
            "Expected transformed molecular descriptors to be "
            "equal to SpectralEmbedding decomposed features",
        )
        remove(csv_fpath)

    def test_invalid_transform_error(self):
        """Using an invalid or unimplemented dimensionality reduction method
        should throw an error.
        """
        n_features = 20
        features = np.random.normal(size=(len(self.test_smiles), n_features))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", feature_arr=features)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            similarity_measure="l0_similarity",
            is_verbose=True,
        )
        features = StandardScaler().fit_transform(features)
        features = TSNE(perplexity=len(self.test_smiles) / 2).fit_transform(features)
        with self.assertRaises(InvalidConfigurationError):
            error_matrix = features - molecule_set.get_transformed_descriptors(
                method_="not a real method"
            )
        remove(csv_fpath)

    def test_clustering_fingerprints(self):
        """
        Test the clustering of molecules featurized by their fingerprints.

        """
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv")
        n_clusters = 3
        for descriptor in SUPPORTED_FPRINTS:
            for similarity_measure in SUPPORTED_SIMILARITIES:
                molecule_set = MoleculeSet(
                    molecule_database_src=csv_fpath,
                    molecule_database_src_type="csv",
                    fingerprint_type=descriptor,
                    similarity_measure=similarity_measure,
                    is_verbose=True,
                )
                with self.assertRaises(NotInitializedError):
                    molecule_set.get_cluster_labels()
                if molecule_set.similarity_measure.is_distance_metric():
                    molecule_set.cluster(n_clusters=n_clusters)
                    self.assertLessEqual(
                        len(set(molecule_set.get_cluster_labels())),
                        n_clusters,
                        "Expected number of cluster labels to be "
                        "less than equal to number of clusters",
                    )
                    if molecule_set.similarity_measure.type_ == "continuous":
                        self.assertEqual(
                            str(molecule_set.clusters_),
                            "kmedoids",
                            f"Expected kmedoids clustering for "
                            f"similarity: {similarity_measure}",
                        )
                    else:
                        self.assertEqual(
                            str(molecule_set.clusters_),
                            "complete_linkage",
                            f"Expected complete_linkage clustering"
                            f"for similarity: {similarity_measure}",
                        )
                else:
                    with self.assertRaises(InvalidConfigurationError):
                        molecule_set.cluster(n_clusters=n_clusters)
        remove(csv_fpath)

    def test_molecule_set_getters(self):
        """Retrieve names and properties of mols using MoleculeSet."""
        properties = np.random.normal(size=len(self.test_smiles))
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", property_seq=properties)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )

        self.assertListEqual(self.test_smiles, molecule_set.get_mol_names().tolist())

        for a, b in zip(
            properties.tolist(), molecule_set.get_mol_properties().tolist()
        ):
            self.assertAlmostEqual(a, b)
        remove(csv_fpath)

    def test_molecule_set_sim_getters(self):
        """Get the properties for most and least similar molecule pairs."""
        properties = np.array([i for i in range(len(self.test_smiles))])
        csv_fpath = self.smiles_seq_to_xl_or_csv(ftype="csv", property_seq=properties)
        molecule_set = MoleculeSet(
            molecule_database_src=csv_fpath,
            molecule_database_src_type="csv",
            fingerprint_type="morgan_fingerprint",
            similarity_measure="tanimoto",
            is_verbose=True,
        )
        _, ref_props = molecule_set.get_property_of_most_dissimilar()
        self.assertListEqual(ref_props, [5, 5, 5, 5, 5, 0, 7, 0])

        _, ref_props = molecule_set.get_property_of_most_similar()
        self.assertListEqual(ref_props, [1, 2, 1, 4, 3, 6, 5, 3])

        remove(csv_fpath)


if __name__ == "__main__":
    unittest.main()
