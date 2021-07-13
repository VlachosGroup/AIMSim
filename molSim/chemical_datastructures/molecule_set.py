"""
Abstraction of a data set comprising multiple Molecule objects.
"""
from glob import glob
import os.path

import multiprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.utils import resample

from molSim.chemical_datastructures import Molecule
from molSim.exceptions import NotInitializedError
from molSim.ops.clustering import Cluster
from molSim.ops.descriptor import Descriptor
from molSim.ops.similarity_measures import SimilarityMeasure


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
    sample_ratio : float
        Fraction of dataset to keep for analysis. Default is 1.

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
                 similarity_measure,
                 n_threads=1,
                 fingerprint_type=None,
                 sampling_ratio=1.,
                 sampling_random_state=42):
        """
        Parameters
        ----------
        sampling_ratio : float
            Fraction of the molecules to keep. Useful for selection subset
            of dataset for quick computations.
        sampling_random_state : int
        Random state used for sampling. Default is 42.
        """
        self.is_verbose = is_verbose
        self.n_threads = n_threads
        self.molecule_database = None
        self.descriptor = Descriptor()
        self._set_molecule_database(molecule_database_src,
                                    molecule_database_src_type)
        if 0. < sampling_ratio < 1.:
            self._subsample_database(sampling_ratio=sampling_ratio,
                                     random_state=sampling_random_state)
        if fingerprint_type is not None:
            # overrides if descriptor set in self._set_molecule_database
            self._set_descriptor(fingerprint_type=fingerprint_type)
        self.similarity_measure = SimilarityMeasure(similarity_measure)
        self.similarity_matrix = None
        self._set_similarity_matrix()
        self.clusters = None

    def _set_molecule_database(self,
                               molecule_database_src,
                               molecule_database_src_type):
        """Load molecular database and set attribute molecule_database.

        Parameters
        ----------
        molecule_database_src : str
            Source of molecular information. Can be a folder or a filepath.
            In case a folder is specified, all .pdb files in the folder
            are sequentially read.
            If a file path, it is assumed that the file is a .txt file with
            layout: SMILES string (column1) '\b' property (column2, optional).
        molecule_database_src_type : str
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
                database_feature_df = database_feature_df.drop(['feature_name'],
                                                               axis=1)
            if 'feature_smiles' in feature_cols:
                mol_smiles = database_df['feature_smiles'].values.flatten()
                database_feature_df = database_feature_df.drop(
                                                             ['feature_smiles'],
                                                             axis=1)

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
            if len(database_feature_df.columns) > 0:
                _set_descriptor(
                    self,
                    arbitrary_descriptor_vals=database_feature_df.values)

        else:
            raise FileNotFoundError(
                f'{molecule_database_src} could not be found. '
                f'Please enter valid folder name or path of a '
                f'text/excel/csv')
        if len(molecule_database) == 0:
            raise UserWarning('No molecular files found in the location!')
        self.molecule_database = molecule_database

    def _subsample_database(self, sampling_ratio, random_state):
        n_samples = int(sampling_ratio * len(self.molecule_database))
        self.molecule_database = resample(self.molecule_database,
                                          replace=False,
                                          n_samples=n_samples,
                                          random_state=random_state)

    def _set_descriptor(self,
                        arbitrary_descriptor_vals=None,
                        fingerprint_type=None):
        """Sets molecule.descriptor attribute for each molecule object in
        MoleculeSet. Either use arbitrary_descriptor_vals to pass descriptor
        values manually or pass fingerprint_type to generate a fingerprint
        from molecule_graph. Both can't be None.

        Parameters
        ----------
        arbitrary_descriptor_vals : np.ndarray
            Arbitrary descriptor array of size:
                (n_mols xx dimensionality of descriptor space).
                Default is None.
        fingerprint_type : str
            String label specifying which fingerprint to use. Default is None.

        """
        for molecule_id, molecule in enumerate(self.molecule_database):
            if fingerprint_type is not None:
                molecule.set_descriptor(fingerprint_type=fingerprint_type)
            elif arbitrary_descriptor_vals is not None:
                molecule.set_descriptor(
                    arbitrary_descriptor_val=arbitrary_descriptor_vals[
                                                                   molecule_id])
            else:
                raise ValueError('No descriptor vector or fingerprint type '
                                 'were passed.')

    def _set_similarity_matrix(self):
        """Calculate the similarity metric using a molecular descriptor
        and a similarity measure. Set this attribute.

        """
        n_mols = len(self.molecule_database)
        similarity_matrix = np.zeros(shape=(n_mols, n_mols))
        
        # Parallel implementation of similarity calculations.
        if self.n_threads > 1:
            m = multiprocess.Manager()
            q = m.Queue()
            # worker thread
            def worker(thread_idx, n_mols, start_idx, end_idx, queue):
                # make a local copy of the overall similarity matrix
                local_similarity_matrix = np.zeros(shape=(n_mols, n_mols))
                if self.is_verbose:
                    print("thread",thread_idx,"will calculate molecules",start_idx,"through",end_idx)
                # same iteration as serial implementation, but only compute source molecules in the specified range
                for source_mol_id, molecule in enumerate(self.molecule_database):
                    if source_mol_id >= start_idx and source_mol_id < end_idx:
                        for target_mol_id in range(source_mol_id, n_mols):
                            if self.is_verbose:
                                print(f'thread {thread_idx} computing similarity of molecule num '
                                    f'{target_mol_id+1} against {source_mol_id+1}')
                            try:
                                similarity_matrix[source_mol_id, target_mol_id] = \
                                    molecule.get_similarity_to_molecule(
                                                self.molecule_database[target_mol_id],
                                                similarity_measure=self.similarity_measure)
                            except NotInitializedError as e:
                                e.message += 'Similarity matrix could not be set '
                                raise e
                            # symmetric matrix entry
                            local_similarity_matrix[target_mol_id, source_mol_id] = \
                                local_similarity_matrix[source_mol_id, target_mol_id]
                queue.put(local_similarity_matrix)
                return None
            
            # calculate work distribution and spawn threads
            remainder = n_mols % (self.n_threads-1)
            bulk = n_mols // (self.n_threads-1)
            threads = []
            for i in range(self.n_threads-1):
                thread = multiprocess.Process(target=worker, args=(i, n_mols, i*bulk, bulk*(i+1), q, ))
                thread.daemon = True
                threads.append(thread)
                thread.start()
            thread = multiprocess.Process(target=worker, args=(self.n_threads-1, n_mols, (self.n_threads-1)*bulk, (self.n_threads-1)*bulk+remainder+1, q, ))
            thread.daemon = True
            threads.append(thread)
            thread.start()

            # retrieve the result and sum all the matrices together.
            for thread in threads:
                thread.join()
            thread_results = []
            for _ in range(self.n_threads-1):
                thread_results.append(q.get())
            similarity_matrix = sum(thread_results)
            print("done")
        else:
            # serial implementation
            for source_mol_id, molecule in enumerate(self.molecule_database):
                for target_mol_id in range(source_mol_id, n_mols):
                    if self.is_verbose:
                        print('Computing similarity of molecule num '
                            f'{target_mol_id+1} against {source_mol_id+1}')
                    similarity_matrix[source_mol_id, target_mol_id] = \
                        molecule.get_similarity_to_molecule(
                                    self.molecule_database[target_mol_id],
                                    similarity_measure=self.similarity_measure)
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
        self.similarity_measure = SimilarityMeasure(metric=similarity_measure)

    def get_most_similar_pairs(self):
        """Get pairs of samples which are most similar.

        Parameters
        ----------


        Returns
        -------
        List(Tuple(Molecule, Molecule))
            List of pairs of Molecules closest to one another.
            Since ties are broken randomly, this may be non-transitive
            i.e. (A, B) =/=> (B, A)

        """
        if self.similarity_matrix is None:
            raise NotInitializedError('MoleculeSet instance not properly '
                                      'initialized with descriptor and '
                                      'similarity measure')
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
                                  descriptor=None,
                                  similarity_measure=None):
        """Get pairs of samples which are least similar.

        Parameters
        ----------

        Returns
        -------
        List(Tuple(Molecule, Molecule))
            List of pairs of indices closest to one another.

        """
        if self.similarity_matrix is None:
            raise NotInitializedError('MoleculeSet instance not properly '
                                      'initialized with descriptor and '
                                      'similarity measure')

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
    
    def get_mol_names(self):
        mol_names = []
        for mol_id, mol in enumerate(self.molecule_database):
            mol_name = mol.get_name()
            if mol_name is None:
                mol_names.append('id: '+ str(mol_id))
            else:
                 mol_names.append(mol_name)

    def cluster(self, n_clusters, clustering_method='kmedoids', **kwargs):
        self.clusters = Cluster(n_clusters=n_clusters, 
                                clustering_method=clustering_method,
                                **kwargs).fit(self.get_distance_matrix())
        mol_names = np.array(self.get_mol_names())
        cluster_grouped_mol_names = {}
        for cluster_id in range(n_clusters):
            cluster_grouped_mol_names[cluster_id] = mol_names[
                                                        self.clusters.get_labels 
                                                        == cluster_id].tolist()
        return cluster_grouped_mol_names

