"""Data structures relevant for storing molecular information
Data Structures Implemented
---------------------------
1. Molecule
    Abstraction of a molecule with relevant property manipulation methods.
2. MoleculeSet
    Abstraction of a data set comprising multiple Molecule objects.
"""
from glob import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs, Chem
from rdkit.Chem import Draw
from sklearn_extra.cluster import KMedoids

from molSim.helper_methods import get_feature_datatype
from molSim.featurize_molecule import Descriptor
import molSim.similarity_measures as similarity_measures


class Molecule:
    """Molecular object defined from RDKIT mol object.

    """
    def __init__(self,
                 mol_graph=None,
                 mol_text=None,
                 mol_property_val=None,
                 mol_descriptor_val=None,
                 mol_src=None,
                 mol_smiles=None):
        """Constructor

        Parameters
        ----------
        mol_graph: RDKIT mol object
            Graph-level information of molecule.
            Implemented as an RDKIT mol object. Default is None.
        mol_text: str
            Text identifier of the molecule. Default is None.
            Identifiers can be:
            ------------------
            1. Name of the molecule.
            2. SMILES string representing the molecule.
        mol_property_val: float
            Some property associated with the molecule. This is typically the
            response being studied. E.g. Boiling point, Selectivity etc.
            Default is None.
        mol_descriptor_val: numpy ndarray
            Decriptor value for the molecule. Must be numpy array or list.
            Default is None.
        mol_src: str
            Source file or SMILES string to load molecule. Acceptable files are
              -> .pdb file
              -> .txt file with SMILE string in first column, first row and
                      (optionally) property in second column, first row.
            Default is None.
            If provided mol_graph is attempted to be loaded from it.
        mol_smiles: str
            SMILES string for molecule. If provided, mol_graph is loaded from
            it. If mol_text not set in keyword argument, this string is used
            to set it.

        """
        self.mol_graph = mol_graph
        self.mol_text = mol_text
        self.mol_property_val = mol_property_val
        self.descriptor = Descriptor() if mol_descriptor_val is None \
            else Descriptor(value=np.array(mol_descriptor_val))
        if mol_src is not None:
            self._set_molecule_from_file(mol_src)
            if self.mol_graph is None:
                raise ValueError('Could not load molecule from file source',
                                 mol_src)
        if mol_smiles is not None:
            self._set_molecule_from_smiles(mol_smiles)
            if self.mol_graph is None:
                raise ValueError('Could not load molecule from SMILES string',
                                 mol_smiles)

    def _set_molecule_from_smiles(self, mol_smiles):
        """
        Set the mol_graph attribute from smiles string.
        If self.mol_text is not set, it is set to the smiles string.

        Parameters
        ----------
        mol_smiles: str
        SMILES string for molecule. If provided, mol_graph is loaded from
            it. If mol_text not set in keyword argument, this string is used
            to set it.

        """
        self.mol_graph = Chem.MolFromSmiles(mol_smiles)
        if self.mol_text is None:
            self.mol_text = mol_smiles

    def _set_molecule_from_file(self, mol_src):
        """Load molecule graph from file

        Parameters
        mol_src: str
            Source file or SMILES string to load molecule.
            Acceptable files are
              -> .pdb file
              -> .txt file with SMILE string in first column, first row.

        """
        if os.path.isfile(mol_src):
            mol_fname, extension = os.path.basename(mol_src).split('.')
            if extension == 'pdb':
                # read pdb file
                self.mol_graph = Chem.MolFromPDBFile(mol_src)
                if self.mol_text is None:
                    self.mol_text = mol_fname
            elif extension == 'txt':
                with open(mol_src, "r") as fp:
                    mol_smiles = fp.readline().split()[0]
                self._set_molecule_from_smiles(mol_smiles)

    def get_similarity_to_molecule(self,
                                   target_mol,
                                   similarity_measure,
                                   molecular_descriptor):
        """Get a similarity metric to a target molecule

        Parameters
        ----------
        target_mol: Molecule object: Target molecule.
            Similarity score is with respect to this molecule
        similarity_measure: str
            The similarity metric used.
        molecular_descriptor : str
            The molecular descriptor used to encode molecules.
            *** Supported Descriptors ***
            'morgan_fingerprint'

        Returns
        -------
        similarity_score: float
            Similarity coefficient by the chosen method.

        """
        feature_datatype = get_feature_datatype(
                                     similarity_measure=similarity_measure,
                                     molecular_descriptor=molecular_descriptor)
        self.descriptor.make_fingerprint(molecule_graph=self.mol_graph,
                                         fingerprint_type=molecular_descriptor,
                                         fingerprint_datatype=feature_datatype)
        target_mol.descriptor.make_fingerprint(
                                         molecule_graph=target_mol.mol_graph,
                                         fingerprint_type=molecular_descriptor,
                                         fingerprint_datatype=feature_datatype)
        if similarity_measure == 'tanimoto':
            return similarity_measures.get_tanimoto_similarity(
                                                         self.descriptor,
                                                         target_mol.descriptor)
        elif similarity_measure == 'neg_l0':
            return similarity_measures.get_l_similarity(self.descriptor,
                                                        target_mol.descriptor,
                                                        order=0)
        elif similarity_measure == 'neg_l1':
            return similarity_measures.get_l_similarity(self.descriptor,
                                                        target_mol.descriptor,
                                                        order=1)
        elif similarity_measure == 'neg_l2':
            return similarity_measures.get_l_similarity(self.descriptor,
                                                        target_mol.descriptor,
                                                        order=2)
        elif similarity_measure == 'dice':
            return similarity_measures.get_dice_similarity(
                                                        self.descriptor,
                                                        target_mol.descriptor)
        else:
            raise ValueError('Similarity measure note specified correctly')

    def compare_to_molecule_set(self, molecule_set):
        """Compare the molecule to a database contained in
        a MoleculeSet object.

        Parameters
        ----------
        molecule_set: MoleculeSet object
            Database of molecules to compare against.

        Returns
        -------
        target_similarity: list
           List of similarity scores of molecules of the database when
           compared to the self molecule.

        Note
        ----
        Excludes the self molecule if it is part of the same database.
        Uses mol_text attribute to achieve this.

        """
        target_similarity = [
            self.get_similarity_to_molecule(
                ref_mol, similarity_measure=molecule_set.similarity_measure,
                molecular_descriptor=molecule_set.molecular_descriptor)
            for ref_mol in molecule_set.molecule_database
            if ref_mol.mol_text != self.mol_text]
        return target_similarity

    def get_mol_property_val(self):
        return self.mol_property_val
    
    def draw(self, fpath=None, **kwargs):
        """Draw or molecule graph.

        Parameters
        ----------
        fpath: str
            Path of file to store image. If None, image is displayed in io.
            Default is None.
        kwargs: keyword arguments
            Arguments to modify plot properties.

        """
        if fpath is None:
            Draw.MolToImage(self.mol_graph, **kwargs).show()
        else:
            Draw.MolToFile(self.mol_graph, fpath, **kwargs)
        

class MoleculeSet:
    """Collection of Molecule objects.

    Attributes
    ----------
    molecule_database: List
        List of Molecule objects.
    similarity_measure : str
        Similarity measure used.
    similarity_matrix: numpy ndarray
        n_mols X n_mols numpy matrix of pairwise similarity scores.
    is_verbose : bool
        Controls how much information is displayed during plotting.

    Methods
    -------
    generate_similarity_matrix()
        Set the similarity_matrix attribute.
    get_most_similar_pairs()
        Get the indexes of the most similar molecules as tuples.

    """
    def __init__(self,
                 molecule_database_src,
                 molecule_database_src_type,
                 is_verbose,
                 similarity_measure=None,
                 molecular_descriptor=None):
        self.is_verbose = is_verbose
        self.molecule_database = None
        self.molecular_descriptor = None
        self.similarity_measure = None
        self.similarity_matrix = None
        self.clusters = []
        
        if molecule_database_src is not None \
            and molecule_database_src_type is not None:
            self._set_molecule_database(molecule_database_src,
                                        molecule_database_src_type)
        if similarity_measure is not None:
            self._set_similarity_measure(similarity_measure)
        if molecular_descriptor is not None:
            self._set_molecular_descriptor(molecular_descriptor)
        if self.molecular_descriptor and self.similarity_measure:
            self._set_similarity_matrix()

    def _set_molecule_database(self,
                               molecule_database_src,
                               molecule_database_src_type):
        """Load molecular database and set as attribute.

        Parameters
        ----------
        molecule_database_src: str
            Source of molecular information. Can be a folder or a filepath.
            In case a folder is specified, all .pdb files in the folder
            are sequentially read.
            If a file path, it is assumed that the file is a .txt file with
            layout: SMILES string (column1) '\b' property (column2, optional).
        molecule_database_src_type: str
            Type of source. Can be ['folder', 'text', 'excel', 'csv']

        Returns
        -------
        None
        Sets self.molecule_database: List(Molecule)

        """
        molecule_database = []
        if molecule_database_src_type.lower() in ['folder', 'directory']:
            if self.is_verbose:
                print(f'Searching for *.pdb files in {molecule_database_src}')
            for molfile in glob(os.path.join(molecule_database_src, '*.pdb')):
                mol_graph = Chem.MolFromPDBFile(molfile)
                mol_text = os.path.basename(molfile).replace('.pdb', '')
                if mol_graph is None:
                    print(f'{molfile} could not be imported. Skipping')
                else:
                    molecule_database.append(Molecule(mol_graph, mol_text))
        elif molecule_database_src_type.lower() == 'text':
            if self.is_verbose:
                print(f'Reading SMILES strings from {molecule_database_src}')
            with open(molecule_database_src, "r") as fp:
                smiles_data = fp.readlines()
            for count, line in enumerate(smiles_data):
                # Assumes that the first column contains the smiles string
                line_fields = line.split()
                smile = line_fields[0]
                mol_property_val = None
                if len(line_fields) > 1:
                    mol_property_val = float(line_fields[1])
                if self.is_verbose:
                    print(f'Processing {smile} '
                          f'({count + 1}/{len(smiles_data)})')
                mol_graph = Chem.MolFromSmiles(smile)
                if mol_graph is None:
                    print(f'{smile} could not be loaded')
                else:
                    mol_text = smile
                    molecule_database.append(
                        Molecule(mol_graph=mol_graph,
                                 mol_text=mol_text,
                                 mol_property_val=mol_property_val))
        elif molecule_database_src_type.lower() in ['excel', 'csv']:
            if self.is_verbose:
                print(f'Reading molecules from {molecule_database_src}')
            database_df = pd.read_excel(molecule_database_src,
                                        engine='openpyxl') \
                if molecule_database_src_type.lower() == 'excel'\
                else pd.read_csv(molecule_database_src)
            # expects feature columns to be prefixed with feature_
            # e.g. feature_smiles
            feature_cols = [column for column in database_df.columns
                            if column.split('_')[0] == 'feature']
            database_feature_df = database_df[feature_cols]
            mol_names, mol_smiles, responses = None, None, None
            if 'feature_name' in feature_cols:
                mol_names = database_feature_df['feature_name'].values.flatten()
            if 'feature_smiles' in feature_cols:
                mol_smiles = database_df['feature_smiles'].values.flatten()

            response_col = [column for column in database_df.columns
                            if column.split('_')[0] == 'response']
            if len(response_col) > 0:
                # currently handles one response
                responses = database_df[response_col].values.flatten()
            for mol_id, smile in enumerate(mol_smiles):
                if self.is_verbose:
                    print(f"Processing {smile} "
                          f"({mol_id + 1}/"
                          f"{database_df['feature_smiles'].values.size})")
                mol_graph = Chem.MolFromSmiles(smile)
                if mol_graph is None:
                    print(f'{smile} could not be loaded')
                else:
                    mol_text = smile
                    mol_property_val = None
                    if mol_names is not None:
                        mol_text = mol_names[mol_id]
                    if responses is not None:
                        mol_property_val = responses[mol_id]
                    molecule_database.append(
                        Molecule(mol_graph=mol_graph,
                                 mol_text=mol_text,
                                 mol_property_val=mol_property_val))

        else:
            raise FileNotFoundError(
                f'{molecule_database_src} could not be found. '
                f'Please enter valid foldername or path of a '
                f'text/excel/csv')
        if len(molecule_database) == 0:
            raise UserWarning('No molecular files found in the location!')
        self.molecule_database = molecule_database

    def _set_molecular_descriptor(self, molecular_descriptor):
        """Sets molecular descriptor attribute.

        Parameters
        ----------
        molecular_descriptor: str
            String label specifying which descriptor to use for featurization.
            See docstring for implemented descriptors and labels.

        """
        if molecular_descriptor not in Descriptor.get_supported_descriptors():
            raise NotImplementedError(f'{molecular_descriptor} '
                                      'is currently not supported')
        self.molecular_descriptor = molecular_descriptor

    def _set_similarity_measure(self, similarity_measure):
        """Set the similarity measure attribute.

        Parameters
        ----------
        similarity_measure: str
            The similarity metric used. See docstring for list
            of supported similarity metrics.

        """
        if similarity_measure not in similarity_measures.get_supported_measures():
            raise NotImplementedError(f'{similarity_measure} '
                                      'is currently not supported')
        self.similarity_measure = similarity_measure
    
    def _set_similarity_matrix(self):
        """Calculate the similarity metric using a molecular descriptor
        and a similarity measure. Set this attribute.

        """
        n_mols = len(self.molecule_database)
        similarity_matrix = np.zeros(shape=(n_mols, n_mols))
        for source_mol_id, molecule in enumerate(self.molecule_database):
            for target_mol_id in range(source_mol_id, n_mols):
                if self.is_verbose:
                    print('Computing similarity of molecule num '
                          f'{target_mol_id+1} against {source_mol_id+1}')
                similarity_matrix[source_mol_id, target_mol_id] = \
                    molecule.get_similarity_to_molecule(
                                self.molecule_database[target_mol_id],
                                similarity_measure=self.similarity_measure,
                                molecular_descriptor=self.molecular_descriptor)
                # symmetric matrix entry
                similarity_matrix[target_mol_id, source_mol_id] = \
                    similarity_matrix[source_mol_id, target_mol_id]
        self.similarity_matrix = similarity_matrix

    def _set_similarity_measure(self, similarity_measure):
        """Set the similarity measure attribute.

        Parameters
        ----------
        similarity_measure: str
            The similarity metric used. See docstring for list
            of supported similarity metrics.

        """
        if similarity_measure not in similarity_measures.get_supported_measures():
            raise NotImplementedError(f'{similarity_measure} '
                                      'is currently not supported')
        self.similarity_measure = similarity_measure

    def get_most_similar_pairs(self,
                               molecular_descriptor=None,
                               similarity_measure=None):
        """Get pairs of samples which are most similar.

        Parameters
        ----------
        molecular_descriptor: str
            If descriptor was not defined for this data set,
            must be defined now. Default is None.
        similarity_measure: str
            If similarity_measure was not defined for this data set,
            must be defined now. Default is None.

        Returns
        -------
        List(Tuple(Molecule, Molecule))
            List of pairs of Molecules closest to one another.
            Since ties are broken randomly, this may be non-transitive
            i.e. (A, B) =/=> (B, A)

        """
        if molecular_descriptor is not None:
            self._set_molecular_descriptor(molecular_descriptor)
            if similarity_measure is not None:
                self._set_similarity_measure(similarity_measure)
                self._set_similarity_matrix()
        if self.molecular_descriptor is None:
            raise ValueError('Feature datatype could not be set, probably'
                             'due to bad molecular_descriptor argument')
        if self.similarity_measure is None:
            raise ValueError('Similarity measure not set')

        n_samples = self.similarity_matrix.shape[0]
        found_samples = [0 for _ in range(n_samples)]
        out_list = []
        for index, row in enumerate(self.similarity_matrix):
            if found_samples[index]:
                # if  species has been identified before
                continue
            post_diag_closest_index = np.argmax(row[(index + 1):]) \
                + index + 1 if index < n_samples-1 else -1
            pre_diag_closest_index = np.argmax(row[:index]) if index > 0 \
                else -1
            # if either (pre_) post_diag_closest_index not set, the
            # closest_index is set to the (post_) pre_diag_closest_index
            if pre_diag_closest_index == -1:
                closest_index = post_diag_closest_index
            if post_diag_closest_index == -1:
                closest_index = pre_diag_closest_index
            # if both pre and post index set, closest_index_index set to index
            # with min distance. In case of tie, post_diag_closest_index set
            else:
                # choose the index which has max correlation
                closest_index = (
                    post_diag_closest_index if
                    row[post_diag_closest_index] >= row[pre_diag_closest_index]
                    else pre_diag_closest_index)
            out_list.append((self.molecule_database[index],
                             self.molecule_database[closest_index]))
            # update list
            found_samples[closest_index] = 1
            found_samples[index] = 1
        return out_list

    def get_most_dissimilar_pairs(self,
                                  molecular_descriptor=None,
                                  similarity_measure=None):
        """Get pairs of samples which are least similar.

        Parameters
        ----------
        molecular_descriptor: str
            If descriptor was not defined for this data set,
            must be defined now. Default is None.
        similarity_measure: str
            If similarity_measure was not defined for this data set,
            must be defined now. Default is None.

        Returns
        -------
        List(Tuple(Molecule, Molecule))
            List of pairs of indices closest to one another.

        """
        if molecular_descriptor is not None:
            self._set_molecular_descriptor(molecular_descriptor)
            if similarity_measure is not None:
                self._set_similarity_measure(similarity_measure)
                self._set_similarity_matrix()

        n_samples = self.similarity_matrix.shape[0]
        found_samples = [0 for _ in range(n_samples)]
        out_list = []
        for index, row in enumerate(self.similarity_matrix):
            if found_samples[index]:
                # if  species has been identified before
                continue
            furthest_index = np.argmin(row)
            out_list.append((self.molecule_database[index],
                             self.molecule_database[furthest_index]))
            # update list
            found_samples[furthest_index] = 1
            found_samples[index] = 1
        return out_list

    def get_similarity_matrix(self):
        """Get the similarity matrix for the data set.

        Returns
        -------
        np.ndarray
            Similarity matrix of the dataset.

        Note
        ----
        If un-set, sets the self.similarity_matrix attribute.

        """
        if self.similarity_matrix is None:
            self._set_similarity_matrix()
        return self.similarity_matrix
    
    def get_distance_matrix(self):
        """Get the distance matrix for the data set defined here as:
        distance_matrix =  -similarity_matrix.

        Returns
        -------
        np.ndarray
            Similarity matrix of the dataset.

        Note
        ----
        If un-set, sets the self.similarity_matrix attribute.

        """
        return -self.get_similarity_matrix()        
    
    def get_pairwise_similarities(self):
        pairwise_similarity_vector = []
        for ref_mol in range(len(self.molecule_database)):
            for target_mol in range(ref_mol+1, len(self.molecule_database)):
                pairwise_similarity_vector.append(
                                    self.similarity_matrix[ref_mol, target_mol])
        return np.array(pairwise_similarity_vector)
    
    def cluster(n_clusters, algorithm='kmedoids', **kwargs):
        if algorithm == 'kmedoids':

            


    
    





