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

import numpy as np
from rdkit import DataStructs, Chem

import featurize_molecule
from similarity_measures import get_supported_measures


class Molecule:
    """Molecular object defined from RDKIT mol object.

    """
    def __init__(self, mol_graph, mol_text=None, mol_property_val=None):
        """Constructor

        Parameters
        ----------
        mol_graph: RDKIT mol object
            Graph-level information of molecule.
            Implemented as an RDKIT mol object.
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

        """
        self.mol_graph = mol_graph
        self.mol_text = mol_text
        self.mol_property_val = mol_property_val
        self.descriptor = None

    def _set_descriptor_value(self, molecular_descriptor, output_datatype):
        """Set a descriptor value for the molecule.

        Parameters
        ----------
        molecular_descriptor: str
            String label specifying which descriptor to use for featurization.
            See docstring for implemented descriptors and labels.
        output_datatype: str
            String label specifying the datatype of the descriptor value.
            Datatypes currently supported are:
            'numpy': np.array
            'rdkit': rdkit.DataStructs.cDataStructs.ExplicitBitVect

        """
        if molecular_descriptor == 'morgan_fingerprint':
            descriptor_val = featurize_molecule.get_morgan_fingerprint(
                                               self.mol_graph,
                                               output_datatype=output_datatype)
        elif molecular_descriptor == 'rdkit_topological':
            descriptor_val = featurize_molecule.get_rdkit_topological_fingerprint(
                                            self.mol_graph,
                                            output_datatype=output_datatype)
        else:
            raise NotImplementedError(f'{molecular_descriptor} not recognized')
        self.descriptor = {
            'value': descriptor_val,
            'molecular_descriptor': molecular_descriptor
        }

    def get_similarity(self, similarity_measure, mol1_descrptr, mol2_descrptr):
        """Expose a suitable method based on similarity_measure.

        Parameters
        ---------
        similarity_measure: str
            Similarity measure to use.
        mol1_descrptr: int
            Descriptor Representation for molecule 1.
        mol2_descrptr: int
            Descriptor Representation for molecule 2.

        Returns
        ------
        float
            Similarity between the two molecules.

        Similarity Measures Implemented
        -------------------------------
        1. tanimoto_similarity:
            Jaccard coefficient between molecular descriptors.
        2. neg_l0:
            Negative L0 norm between molecular descriptors.
        3. neg_l1:
            Negative L1 norm between molecular descriptors.
        4. neg_l2:
            Negative L2 norm between molecular descriptors.

        """
        if similarity_measure == 'tanimoto_similarity':
            return DataStructs.TanimotoSimilarity(mol1_descrptr, mol2_descrptr)
        elif similarity_measure == 'neg_l0':
            return -np.linalg.norm(np.asarray(mol1_descrptr) - np.asarray(
                mol2_descrptr), ord=0)
        elif similarity_measure == 'neg_l1':
            return -np.linalg.norm(np.asarray(mol1_descrptr) - np.asarray(
                mol2_descrptr), ord=1)
        elif similarity_measure == 'neg_l2':
            return -np.linalg.norm(np.asarray(mol1_descrptr) - np.asarray(
                mol2_descrptr), ord=2)

    def get_similarity_to_molecule(
            self, target_mol, similarity_measure='tanimoto',
            molecular_descriptor='morgan_fingerprint'):
        """Get a similarity metric to a target molecule

        Parameters
        ----------
        target_mol: Molecule object: Target molecule.
            Similarity score is with respect to this molecule
        similarity_measure: str
            The similarity metric used.
            *** Supported Metrics ***
            'tanimoto': Jaccard Coefficient/ Tanimoto Similarity
                    0 (not similar at all) to 1 (identical)
            'neg_l0': Negative L0 norm of |x1 - x2|
            'neg_l1': Negative L1 norm of |x1 - x2|
            'neg_l2': Negative L2 norm of |x1 - x2|
        molecular_descriptor : str
            The molecular descriptor used to encode molecules.
            *** Supported Descriptors ***
            'morgan_fingerprint'

        Returns
        -------
        similarity_score: float
            Similarity coefficient by the chosen method.

        """

        similarity_score = self.get_similarity(
            similarity_measure,
            self.get_molecular_descriptor(molecular_descriptor),
            target_mol.get_molecular_descriptor(molecular_descriptor))

        return similarity_score


class MoleculeSet:
    """Collection of Molecule objects.

    Attributes
    ----------
    molecule_database: List
        List of Molecule objects.
    feature_datatype: str
        Label deciding format for storing descriptor values. See docstring
        for supported formats.
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
                 is_verbose,
                 similarity_measure=None,
                 molecular_descriptor=None):
        self.molecule_database = None
        self._set_molecule_database(molecule_database_src)
        self.is_verbose = is_verbose
        self.similarity_measure = None
        self._set_similarity_measure(similarity_measure=similarity_measure)
        self.feature_datatype = None
        if molecular_descriptor is not None:
            self._set_feature_datatype(molecular_descriptor)
            if similarity_measure is not None:
                self._set_similarity_matrix(molecular_descriptor,
                                            similarity_measure)

    def _set_molecule_database(self, molecule_database_src):
        """Load molecular database and set as attribute.

        Parameters
        ----------
        molecule_database_src: str
            Source of molecular information. Can be a folder or a filepath.
            In case a folder is specified, all .pdb files in the folder
            are sequentially read.
            If a file path, it is assumed that the file is a .txt file with
            layout: SMILES string (column1) property (column2, optional).

        Returns
        -------
        None
        Sets self.molecule_database: List(Molecule)

        """
        molecule_database = []
        if os.path.isdir(molecule_database_src):
            if self.is_verbose:
                print(f'Searching for *.pdb files in {molecule_database_src}')
            for molfile in glob(os.path.join(molecule_database_src, '*.pdb')):
                mol_graph = Chem.MolFromPDBFile(molfile)
                mol_text = os.path.basename(molfile).replace('.pdb', '')
                if mol_graph is None:
                    print(f'{molfile} could not be imported. Skipping')
                    continue
                molecule_database.append(Molecule(mol_graph, mol_text))
        elif os.path.isfile(molecule_database_src):
            if self.is_verbose:
                print(f'Reading SMILES strings from {molecule_database_src}')
            with open(molecule_database_src, "r") as fp:
                smiles_data = fp.readlines()
            for count, line in enumerate(smiles_data):
                # Assumes that the first column contains the smiles string
                smile = line.split()[0]
                if self.is_verbose:
                    print(
                        f'Processing {smile} '
                        f'({count + 1}/{len(smiles_data)})')
                mol_graph = Chem.MolFromSmiles(smile)
                if mol_graph is None:
                    print(f'{smile} could not be loaded')
                    continue
                mol_text = smile
                molecule_database.append(Molecule(mol_graph, mol_text))
        else:
            raise FileNotFoundError(
                f'{molecule_database_src} could not be found. '
                f'Please enter valid foldername or path of a '
                f'text file with SMILES strings')
        if len(molecule_database) == 0:
            raise UserWarning('No molecular files found in the location!')
        self.molecule_database = molecule_database

    def _set_feature_datatype(self, molecular_descriptor):
        """Set the attribute self.feature_datatype based on rules
        conditional on similarity measure and molecular_descriptor

        Parameters
        ----------
        molecular_descriptor: str
            String label specifying which descriptor to use for featurization.
            See docstring for implemented descriptors and labels.

        """
        if self.similarity_measure == 'tanimoto':
            if molecular_descriptor in ['topological_fingerprint',
                                         'morgan_fingerprint',
                                         ]:
                self.feature_datatype = 'rdkit'
        elif self.similarity_measure in ['-l0', '-l1', '-l2']:
            if molecular_descriptor in ['topological_fingerprint',
                                         'morgan_fingerprint',
                                         ]:
                self.feature_datatype = 'numpy'
        else:
            raise NotImplementedError(f'{self.similarity_measure} similarity'
                                      'does not work with '
                                      f'{molecular_descriptor}')

    def _set_similarity_matrix(self, molecular_descriptor, similarity_measure):
        """Calculate the similarity metric using a molecular descriptor
        and a similarity measure. Set this attribute.

        Parameters
        ----------
        molecular_descriptor: str
            String label specifying which descriptor used for featurization.
            See docstring for implemented descriptors and labels.
        similarity_measure: str
            The similarity metric used. See docstring for list
            of supported similarity metrics.

        """
        n_mols = len(self.molecule_database)
        self.similarity_matrix = np.zeros(shape=(n_mols, n_mols))
        for source_mol_id, molecule in enumerate(self.molecule_database):
            for target_mol_id in range(source_mol_id, n_mols):
                if self.is_verbose:
                    print('Computing similarity of molecule num '
                          f'{target_mol_id+1} against {source_mol_id+1}')
                self.similarity_matrix[source_mol_id, target_mol_id] = \
                    molecule.get_similarity_to_molecule(
                                     self.molecule_database[target_mol_id],
                                     similarity_measure=similarity_measure,
                                     molecular_descriptor=molecular_descriptor)
                # symmetric matrix entry
                self.similarity_matrix[target_mol_id, source_mol_id] = \
                    self.similarity_matrix[source_mol_id, target_mol_id]

    def _set_similarity_measure(self, similarity_measure):
        """Set the similarity measure attribute.

        Parameters
        ----------
        similarity_measure: str
            The similarity metric used. See docstring for list
            of supported similarity metrics.

        """
        if similarity_measure not in get_supported_measures():
            raise NotImplementedError(f'{similarity_measure} '
                                      'is currently not supported')
        self.similarity_measure = similarity_measure

    def get_most_similar_pairs(self, **kwargs):
        """Get pairs of samples which are most similar.

        Parameters
        ----------
        kwargs: dict
            Important keyword arguments:
            'similarity_measure': str
                If similarity_measure was not defined for this data set,
                must be defined now.
            'molecular_descriptor': str
                If descriptor was not defined for this data set,
                must be defined now.
        Returns
        -------
        List(Tuple(Molecule, Molecule))
            List of pairs of indices closest to one another.
            Since ties are broken randomly, this may be non-transitive
            i.e. (A, B) =/=> (B, A)

        """

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
            # closest_index_index is set to the (post_) pre_diag_closest_index
            if pre_diag_closest_index == -1:
                closest_index_index = post_diag_closest_index
            if post_diag_closest_index == -1:
                closest_index_index = pre_diag_closest_index
            # if both pre and post index set, closest_index_index set to index
            # with min distance. In case of tie, post_diag_closest_index set
            else:
                # choose the index which has max correlation
                closest_index_index = (
                    post_diag_closest_index if
                    row[post_diag_closest_index] >= row[pre_diag_closest_index]
                    else pre_diag_closest_index)
            out_list.append((index, closest_index_index))
            # update list
            found_samples[closest_index_index] = 1
            found_samples[index] = 1
        return out_list

    def get_most_dissimilar_pairs(self):
        """Get pairs of samples which are least similar.

        Returns
        -------
        List(Tuple(int, int))
            List of pairs of indices closest to one another.

        """
        # If not set, set similarity_matrix.
        if self.similarity_matrix is None:
            self.generate_similarity_matrix()

        n_samples = self.similarity_matrix.shape[0]
        found_samples = [0 for _ in range(n_samples)]
        out_list = []
        for index, row in enumerate(self.similarity_matrix):
            if found_samples[index]:
                # if  species has been identified before
                continue
            furthest_index = np.argmin(row)
            out_list.append((index, furthest_index))
            # update list
            found_samples[furthest_index] = 1
            found_samples[index] = 1
        return out_list