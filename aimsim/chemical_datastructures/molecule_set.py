"""Abstraction of a data set comprising multiple Molecule objects."""
from glob import glob
import psutil
import warnings
import os.path
import multiprocess
import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from aimsim.chemical_datastructures import Molecule
from aimsim.exceptions import *
from aimsim.ops.clustering import Cluster
from aimsim.ops.descriptor import Descriptor
from aimsim.ops.similarity_measures import SimilarityMeasure


class MoleculeSet:
    """An abstraction of a collection of molecules constituting a chemical
    dataset.

    Attributes:
        is_verbose (bool): Controls how much information is displayed during
                plotting.
        molecule_database (list): Collection of Molecule objects.
        descriptor (Descriptor): Descriptor or fingerprint used to featurize
            molecules in the molecule set.
        similarity_measure (SimilarityMeasure): Similarity measure used.
        similarity_matrix (numpy ndarray): n_mols X n_mols matrix of
            pairwise similarity scores.
        sampling_ratio (float): Fraction of dataset to keep for analysis.
            Default is 1.
        n_threads (int or str): Number of threads used for analysis. Can be
           an integer denoting the number of threads or 'auto' to
           heuristically determine if multiprocessing is worthwhile
           based on a curve fitted to the speedup data in the manuscript SI
           Default is 1.

    Methods:
        is_present(target_molecule): Searches the name of a target
            molecule in the molecule set to determine if the target
            molecule is present in the molecule set.
        compare_against_molecule(query_molecule): Compare the a query
            molecule to all molecules of the set.
        get_most_similar_pairs(): Get pairs of samples which are
            most similar.
        get_most_dissimilar_pairs(): Get pairs of samples which are
            least similar.
        get_property_of_most_similar(): Get property of pairs of molecules
            which are most similar to each other.
        get_property_of_most_dissimilar(): Get property of pairs of
            molecule which are most dissimilar to each other.
        get_similarity_matrix(): Get the similarity matrix for the data set.
        get_distance_matrix(): Get the distance matrix for the data set.
            This is can only be done for similarity measures which yields
            valid distances.
        get_pairwise_similarities(): Get an array of pairwise similarities
            of molecules in the set.
        get_mol_names(): Get names of the molecules in the set.
        get_mol_properties(): Get properties of all the molecules
            in the dataset.
        cluster(n_clusters=8, clustering_method=None, **kwargs): Cluster
            the molecules of the MoleculeSet. Implemented methods.
                'kmedoids': for the K-Medoids algorithm.
                'complete_linkage', 'complete':
                    Complete linkage agglomerative hierarchical
                    clustering.
                'average_linkage', 'average':
                    average linkage agglomerative hierarchical clustering.
                'single_linkage', 'single':
                    single linkage agglomerative hierarchical clustering.
                'ward':
                    for Ward's algorithm.
        get_cluster_labels(): Get cluster membership of Molecules.
        get_transformed_descriptors(method_="pca", **kwargs): Use an
            embedding method to transform molecular descriptor to a
            low dimensional representation. Implemented methods are
            Principal Component Analysis ('pca'),
            Multidimensional scaling ('mds'),
            t-SNE ('tsne'), Isomap ('isomap'),
            Spectral Embedding ('spectral_embedding')

    """

    def __init__(
        self,
        molecule_database_src: str,
        molecule_database_src_type: str,
        is_verbose: bool,
        similarity_measure: str,
        n_threads=1,
        fingerprint_type=None,
        fingerprint_params=None,
        sampling_ratio=1.0,
        sampling_random_state=42,
    ):
        """Constructor for the MoleculeSet class.
        Args:
            sampling_ratio (float): Fraction of the molecules to keep. Useful
                for selection subset of dataset for quick computations.
            sampling_random_state (int): Random state used for sampling.
                Default is 42.

        """
        self.is_verbose = is_verbose
        if type(self.is_verbose) is int and self.is_verbose > 1:
            warnings.warn("You have enabled debug-level logging (is_verbose>=2).")
        self.molecule_database = None
        self.descriptor = Descriptor()
        self.molecule_database, descriptors = self._get_molecule_database(
            molecule_database_src, molecule_database_src_type
        )
        if descriptors is not None:
            self._set_descriptor(arbitrary_descriptor_vals=descriptors)
        if 0.0 < sampling_ratio < 1.0:
            if self.is_verbose:
                print(f"Using {int(sampling_ratio * 100)}% of the database...")
            self._subsample_database(
                sampling_ratio=sampling_ratio, random_state=sampling_random_state
            )
        if fingerprint_type is not None:
            if descriptors is not None and is_verbose:
                print(
                    "Descriptor and fingerprint specified."
                    "Descriptors imported from database source will "
                    "be overwritten by fingerprint."
                )
            self._set_descriptor(
                fingerprint_type=fingerprint_type, fingerprint_params=fingerprint_params
            )
        self.similarity_measure = SimilarityMeasure(similarity_measure)
        if n_threads == "auto":

            def speedup_eqn(n_mols, n_procs):
                return 1.8505e-4 * n_mols + 2.235e-1 * n_procs + 7.082e-2

            n_cores = psutil.cpu_count(logical=False)
            n_mols = len(self.molecule_database)
            if speedup_eqn(n_mols, n_cores) > 1.0:
                self.n_threads = n_cores
            elif speedup_eqn(n_mols, n_cores // 2) > 1.0:
                self.n_threads = n_cores // 2
            else:
                self.n_threads = n_cores
        else:
            self.n_threads = n_threads
        self.similarity_matrix = None
        self._set_similarity_matrix()

    def _get_molecule_database(self, molecule_database_src, molecule_database_src_type):
        """Load molecular database and return it.
        Optionally return features if found in excel / csv file.

        Args:
            molecule_database_src (str):
                Source of molecular information. Can be a folder or a filepath.
                In case a folder is specified, all .pdb files in the folder
                are sequentially read.
                If a file path, it is assumed that the file is a .txt file with
                layout: SMILES string (column1) '\b' property (column2, optional).
            molecule_database_src_type (str):
                Type of source. Can be ['folder', 'text', 'excel', 'csv']

        Returns:
            (list(Molecule), np.ndarray or None)
                Returns a tuple. First element of tuple is the molecule_database.
                Second element is array of features of shape
                (len(molecule_database), n_features) or None if None found.

        """
        if not self.is_verbose:
            RDLogger.DisableLog("rdApp.*")

        molecule_database = []
        descriptors = None
        if molecule_database_src_type.lower() in ["folder", "directory"]:
            if self.is_verbose:
                print(f"Searching for *.pdb files in {molecule_database_src}")
            for molfile in glob(os.path.join(molecule_database_src, "*.pdb")):
                if self.is_verbose:
                    print(f"Loading {molfile}")
                try:
                    molecule_database.append(Molecule(mol_src=molfile))
                except LoadingError as e:
                    if self.is_verbose:
                        print(f"{molfile} could not be imported. Skipping")

        elif molecule_database_src_type.lower() == "text":
            if self.is_verbose:
                print(f"Reading SMILES strings from {molecule_database_src}")
            with open(molecule_database_src, "r") as fp:
                smiles_data = fp.readlines()
            for count, line in enumerate(smiles_data):
                # Assumes that the first column contains the smiles string
                line_fields = line.split()
                smile = line_fields[0]
                mol_property_val = None
                if len(line_fields) > 1:
                    mol_property_val = float(line_fields[1])
                if type(self.is_verbose) is int and self.is_verbose > 1:
                    print(
                        f"Processing {smile} " f"({count + 1}/" f"{len(smiles_data)})"
                    )
                mol_text = smile
                try:
                    molecule_database.append(
                        Molecule(
                            mol_smiles=smile,
                            mol_text=mol_text,
                            mol_property_val=mol_property_val,
                        )
                    )
                except LoadingError as e:
                    if self.is_verbose:
                        print(f"{smile} could not be imported. Skipping")

        elif molecule_database_src_type.lower() in ["excel", "csv"]:
            if self.is_verbose:
                print(f"Reading molecules from {molecule_database_src}")
            database_df = (
                pd.read_excel(molecule_database_src, engine="openpyxl")
                if molecule_database_src_type.lower() == "excel"
                else pd.read_csv(molecule_database_src)
            )
            # expects descriptor columns to be prefixed with descriptor_
            # e.g. descriptor_smiles
            descriptor_cols = [
                column
                for column in database_df.columns
                if column.split("_")[0] == "descriptor"
            ]
            database_descriptor_df = database_df[descriptor_cols]
            mol_names, mol_smiles, responses = None, None, None
            if "descriptor_name" in descriptor_cols:
                mol_names = database_descriptor_df["descriptor_name"].values.flatten()
                database_descriptor_df = database_descriptor_df.drop(
                    ["descriptor_name"], axis=1
                )
            if "descriptor_smiles" in descriptor_cols:
                mol_smiles = database_df["descriptor_smiles"].values.flatten()
                database_descriptor_df = database_descriptor_df.drop(
                    ["descriptor_smiles"], axis=1
                )

            response_col = [
                column
                for column in database_df.columns
                if column.split("_")[0] == "response"
            ]
            if len(response_col) > 0:
                # currently handles one response
                responses = database_df[response_col].values.flatten()
            for mol_id in database_descriptor_df.index:
                if type(self.is_verbose) is int and self.is_verbose > 1:
                    print(
                        f"Processing "
                        f"({mol_id + 1}/"
                        f"{len(database_descriptor_df.index)})"
                    )
                mol_smile = mol_smiles[mol_id] if mol_smiles is not None else None
                mol_text = mol_names[mol_id] if mol_names is not None else mol_smile
                mol_property_val = responses[mol_id] if responses is not None else None

                try:
                    molecule_database.append(
                        Molecule(
                            mol_smiles=mol_smile,
                            mol_text=mol_text,
                            mol_property_val=mol_property_val,
                        )
                    )
                except LoadingError as e:
                    if self.is_verbose:
                        print(
                            f"Molecule index {mol_id} could not be imported. "
                            f"Skipping"
                        )

            if len(database_descriptor_df.columns) > 0:
                descriptors = database_descriptor_df.values
        else:
            raise FileNotFoundError(
                f"{molecule_database_src} could not be found. "
                f"Please enter valid folder name or path of a "
                f"text/excel/csv"
            )
        if len(molecule_database) == 0:
            raise UserWarning("No molecular files found in the location!")
        return molecule_database, descriptors

    def _subsample_database(self, sampling_ratio, random_state):
        """Subsample a fixed proportion of the set.

        Args:
            sampling_ratio (float): Proportion of the set.
            random_state (int): Seed for random number generator
                used in sampling.

        """
        n_samples = int(sampling_ratio * len(self.molecule_database))
        self.molecule_database = resample(
            self.molecule_database,
            replace=False,
            n_samples=n_samples,
            random_state=random_state,
        )

    def _set_descriptor(
        self,
        arbitrary_descriptor_vals=None,
        fingerprint_type=None,
        fingerprint_params=None,
    ):
        """Sets molecule.descriptor attribute for each molecule object in
        MoleculeSet. Either use arbitrary_descriptor_vals to pass descriptor
        values manually or pass fingerprint_type to generate a fingerprint
        from molecule_graph. Both can't be None.

        Args:
            arbitrary_descriptor_vals (np.ndarray):
                Arbitrary descriptor array of size:
                    (n_mols xx dimensionality of descriptor space).
                    Default is None.
            fingerprint_type (str):  String label specifying which fingerprint
                to use. Default is None.
            fingerprint_params (dict): Parameters to modify the fingerprint
                generated. Default is None.

        """
        for molecule_id, molecule in enumerate(self.molecule_database):
            if fingerprint_type is not None:
                molecule.set_descriptor(
                    fingerprint_type=fingerprint_type,
                    fingerprint_params=fingerprint_params,
                )
            elif arbitrary_descriptor_vals is not None:
                molecule.set_descriptor(
                    arbitrary_descriptor_val=arbitrary_descriptor_vals[molecule_id]
                )
            else:
                raise ValueError(
                    "No descriptor vector or fingerprint type were passed."
                )

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

            def worker(
                thread_idx, n_mols, start_idx, end_idx, queue
            ):  # pragma: no cover
                # make a local copy of the overall similarity matrix
                local_similarity_matrix = np.zeros(shape=(n_mols, n_mols))
                if type(self.is_verbose) is int and self.is_verbose > 1:
                    print(
                        "thread",
                        thread_idx,
                        "will calculate molecules",
                        start_idx,
                        "through",
                        end_idx,
                        "(",
                        end_idx - start_idx,
                        "total)",
                    )
                # same iteration as serial implementation, but only compute
                # source molecules in the specified range
                for source_mol_id, molecule in enumerate(self.molecule_database):
                    if source_mol_id >= start_idx and source_mol_id < end_idx:
                        for target_mol_id in range(0, n_mols):
                            if type(self.is_verbose) is int and self.is_verbose > 1:
                                print(
                                    f"thread {thread_idx} computing similarity "
                                    f"of molecule num "
                                    f"{target_mol_id + 1} "
                                    f"against {source_mol_id + 1}"
                                )
                            # diagonal entry
                            if target_mol_id == source_mol_id:
                                local_similarity_matrix[
                                    source_mol_id, target_mol_id
                                ] = 1
                            else:  # non-diagonal entries
                                try:
                                    local_similarity_matrix[
                                        source_mol_id, target_mol_id
                                    ] = molecule.get_similarity_to(
                                        self.molecule_database[target_mol_id],
                                        similarity_measure=self.similarity_measure,
                                    )
                                except NotInitializedError as e:
                                    e.message += "Similarity matrix could not be set "
                                    raise e
                                except ValueError as e:
                                    raise RuntimeError(
                                        f"Unable to proccess molecule {molecule.mol_text}"
                                    ) from e
                queue.put(local_similarity_matrix)
                return None

            # calculate work distribution and spawn threads
            remainder = n_mols % (self.n_threads)
            bulk = n_mols // (self.n_threads)
            threads = []
            for i in range(int(self.n_threads)):
                # last thread
                if i == self.n_threads - 1:
                    thread = multiprocess.Process(
                        target=worker,
                        args=(
                            i,
                            n_mols,
                            i * bulk,
                            bulk * (i + 1) + remainder,
                            q,
                        ),
                    )
                    threads.append(thread)
                    thread.start()
                else:
                    thread = multiprocess.Process(
                        target=worker,
                        args=(
                            i,
                            n_mols,
                            i * bulk,
                            bulk * (i + 1),
                            q,
                        ),
                    )
                    threads.append(thread)
                    thread.start()

            # retrieve the result and sum all the matrices together.
            for thread in threads:
                thread.join()
            thread_results = []
            for _ in range(int(self.n_threads)):
                thread_results.append(q.get())
            similarity_matrix = sum(thread_results)
        else:
            # serial implementation
            for source_mol_id, molecule in enumerate(self.molecule_database):
                for target_mol_id in range(n_mols):
                    if type(self.is_verbose) is int and self.is_verbose > 1:
                        print(
                            "Computing similarity of molecule num "
                            f"{target_mol_id + 1} against {source_mol_id + 1}"
                        )
                    try:
                        similarity_matrix[
                            source_mol_id, target_mol_id
                        ] = molecule.get_similarity_to(
                            self.molecule_database[target_mol_id],
                            similarity_measure=self.similarity_measure,
                        )
                    except ValueError as e:
                        raise RuntimeError(
                            f"Unable to proccess molecule {molecule.mol_text}"
                        ) from e

        self.similarity_matrix = similarity_matrix

    def _set_similarity_measure(self, similarity_measure):
        """Set the similarity measure attribute.

        Args:
            similarity_measure (str): The similarity metric used. See
            docstring for list of supported similarity metrics.

        """
        self.similarity_measure = SimilarityMeasure(metric=similarity_measure)

    def _do_pca(self, get_component_info=False, **kwargs):
        """Do principal component analysis (PCA) of the set [1].

        Args:
            get_component_info (bool): If set to true, more detailed
                information about the embedding process is returned.
                Default is False.
            kwargs (dict): Keyword arguments to modify the behaviour of
                the respective embedding methods. See the documentation pages
                listed below for these arguments.
                'pca': https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                ('n_components' is defaulted to 2).

        Returns:
            X (np.ndarray): Transformed embedding of shape
                (n_samples, n_components)
            component_info (dict): More detailed information about the
                embedding_process. Optionally returned if 'get_component_info'
                is set to True.
                keys:
                    "components_",
                    "explained_variance_",
                    "explained_variance_ratio_",
                    "singular_values_"

        References:
            [1] Bishop, C. M., Pattern recognition and machine learning. 2006.

        """
        params = {
            "n_components": kwargs.get("n_components", 2),
            "copy": kwargs.get("copy", True),
            "whiten": kwargs.get("whiten", False),
            "svd_solver": kwargs.get("svd_solver", "auto"),
            "tol": kwargs.get("tol", 0.0),
            "iterated_power": kwargs.get("iterated_power", "auto"),
            "random_state": kwargs.get("random_state", None),
        }
        pca = PCA(**params)
        X = np.array(
            [molecule.get_descriptor_val() for molecule in self.molecule_database]
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = pca.fit_transform(X)
        if not get_component_info:
            return X
        else:
            component_info = {
                "components_": pca.components_,
                "explained_variance_": pca.explain_variance_,
                "explained_variance_ratio_": pca.explained_variance_ratio_,
                "singular_values_": pca.singular_values_,
            }
            return X, component_info

    def _do_mds(self, get_component_info=False, **kwargs):
        """Do multidimensional scaling (mds) of the set [1-3].

        Args:
            get_component_info (bool): If set to true, more detailed
                information about the embedding process is returned.
                Default is False.
            kwargs (dict): Keyword arguments to modify the behaviour of
                the respective embedding methods. See the documentation pages
                listed below for these arguments.
                'mds':  https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

        Returns:
            X (np.ndarray): Transformed embedding of shape
                (n_samples, n_components)
            component_info (dict): More detailed information about the
                embedding_process. Optionally returned if 'get_component_info'
                is set to True.
                keys:
                    "stress_",
                    "n_iter_"

        References:
            [1] Borg, I. and P. Groenen, Modern Multidimensional Scaling:
                Theory and Applications (Springer Series in Statistics). 2005.
            [2] Kruskal, J., Nonmetric multidimensional scaling:
                A numerical method. Psychometrika, 1964. 29(2): p. 115-129.
            [3]	Kruskal, J., Multidimensional scaling by optimizing goodness
                of fit to a nonmetric hypothesis. Psychometrika, 1964.
                29: p. 1-27.

        """
        params = {
            "n_components": kwargs.get("n_components", 2),
            "metric": kwargs.get("metric", True),
            "n_init": kwargs.get("n_init", 4),
            "max_iter": kwargs.get("max_iter", 3000),
            "verbose": kwargs.get("verbose", 0),
            "eps": kwargs.get("eps", 1e-3),
            "random_state": kwargs.get("random_state", 42),
        }
        embedding = MDS(dissimilarity="precomputed", **params)
        dissimilarity_matrix = self.get_distance_matrix()
        X = embedding.fit_transform(dissimilarity_matrix)
        if not get_component_info:
            return X
        else:
            component_info = {
                "stress_": embedding.stress_,
                "n_iter_": embedding.n_iter_,
            }
            return X, component_info

    def _do_tsne(self, get_component_info=False, **kwargs):
        """Do t-SNE (tsne) of the set [1].

        Args:
            get_component_info (bool): If set to true, more detailed
                information about the embedding process is returned.
                Default is False.
            kwargs (dict): Keyword arguments to modify the behaviour of
                the respective embedding methods. See the documentation pages
                listed below for these arguments.
                'tsne': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

        Returns:
            X (np.ndarray): Transformed embedding of shape
                (n_samples, n_components)
            component_info (dict): More detailed information about the
                embedding_process. Optionally returned if 'get_component_info'
                is set to True.
                keys:
                    "kl_divergence",
                    "n_iter_"
        References:
            [1] van der Maaten, L. and G. Hinton, Viualizing data using t-SNE.
                Journal of Machine Learning Research, 2008. 9: p. 2579-2605.

        """
        params = {
            "n_components": kwargs.get("n_components", 2),
            "perplexity": kwargs.get("perplexity", 30),
            "early_exaggeration": kwargs.get("early_exaggeration", 12),
            "learning_rate": kwargs.get("learning_rate", 200),
            "n_iter": kwargs.get("n_iter", 1000),
            "n_iter_without_progress": kwargs.get("n_iter_without_progress", 300),
            "min_grad_norm": kwargs.get("min_grad_norm", 1e-7),
            "init": kwargs.get("init", "random"),
            "verbose": kwargs.get("verbose", 0),
            "method": kwargs.get("method", "barnes_hut"),
            "angle": kwargs.get("angle", 0.5),
            "n_jobs": kwargs.get("n_jobs", None),
        }
        embedding = TSNE(metric="precomputed", **params)
        dissimilarity_matrix = self.get_distance_matrix()
        X = embedding.fit_transform(dissimilarity_matrix)
        if not get_component_info:
            return X
        else:
            component_info = {
                "kl_divergence": embedding.kl_divergence_,
                "n_iter_": embedding.n_iter_,
            }
            return X, component_info

    def _do_isomap(self, get_component_info=False, **kwargs):
        """Do Isomap (isomap) of the set [1].

        Args:
            get_component_info (bool): If set to true, more detailed
                information about the embedding process is returned.
                Default is False.
            kwargs (dict): Keyword arguments to modify the behaviour of
                the respective embedding methods. See the documentation pages
                listed below for these arguments.
                'isomap': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html

        Returns:
            X (np.ndarray): Transformed embedding of shape
                (n_samples, n_components)
            component_info (dict): More detailed information about the
                embedding_process. Optionally returned if 'get_component_info'
                is set to True.
                keys:
                    "kernel_pca_",
                    "nbrs_"
        References:
            [1] Tenenbaum, J.B., V.d. Silva, and J.C. Langford,
                A Global Geometric Framework for Nonlinear Dimensionality
                Reduction. Science, 2000. 290(5500): p. 2319-2323.

        """
        params = {
            "n_neighbors": kwargs.get("n_neighbors", 5),
            "n_components": kwargs.get("n_components", 2),
            "eigen_solver": kwargs.get("eigen_solver", "auto"),
            "tol": kwargs.get("tol", 0),
            "max_iter": kwargs.get("max_iter", None),
            "path_method": kwargs.get("path_method", "auto"),
            "neighbors_algorithm": kwargs.get("neighbors_algorithm", "auto"),
            "n_jobs": kwargs.get("n_jobs", None),
            "p": kwargs.get("p", 2),
        }
        embedding = Isomap(metric="precomputed", **params)
        dissimilarity_matrix = self.get_distance_matrix()
        X = embedding.fit_transform(dissimilarity_matrix)
        if not get_component_info:
            return X
        else:
            component_info = {
                "kernel_pca_": embedding.kernel_pca_,
                "nbrs_": embedding.nbrs_,
            }
            return X, component_info

    def _do_spectral_embedding(self, get_component_info=False, **kwargs):
        """Do Spectral Embedding (spectral_embedding) of the set [1].

        Args:
            get_component_info (bool): If set to true, more detailed
                information about the embedding process is returned.
                Default is False.
            kwargs (dict): Keyword arguments to modify the behaviour of
                the respective embedding methods. See the documentation pages
                listed below for these arguments.
                'spectral_embedding': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html

        Returns:
            X (np.ndarray): Transformed embedding of shape
                (n_samples, n_components)
            component_info (dict): More detailed information about the
                embedding_process. Optionally returned if 'get_component_info'
                is set to True.
                keys:
                    "n_neighbors_"

        References:
            [1] Ng, A.Y., M.I. Jordan, and Y. Weiss. On Spectral Clustering:
                Analysis and an algorithm. 2001. MIT Press.

        """
        params = {
            "n_components": kwargs.get("n_components", 2),
            "gamma": kwargs.get("gamma", None),
            "random_state": kwargs.get("random_state", None),
            "eigen_solver": kwargs.get("eigen_solver", None),
            "n_neighbors": kwargs.get("n_neighbors", None),
            "n_jobs": kwargs.get("n_jobs", None),
        }
        embedding = SpectralEmbedding(affinity="precomputed", **params)
        similarity_matrix = self.get_similarity_matrix()
        X = embedding.fit_transform(similarity_matrix)
        if not get_component_info:
            return X
        else:
            component_info = {"n_neighbors_": embedding.n_neighbors_}
            return X, component_info

    def is_present(self, target_molecule):
        """
        Searches the name of a target molecule in the molecule set to
        determine if the target molecule is present in the molecule set.

        Args:
            target_molecule (AIMSim.chemical_datastructures.Molecule):
                Target molecule to search.

        Returns:
            (bool): If the molecule is present in the molecule set or not.

        """
        for set_molecule in self.molecule_database:
            if Molecule().is_same(set_molecule, target_molecule):
                return True
        return False

    def compare_against_molecule(self, query_molecule):
        """
        Compare the a query molecule to all molecules of the set.

        Args:
            query_molecule (AIMSim.chemical_datastructures Molecule): Target
                molecule to compare.

        Returns:
            set_similarity (np.ndarray): Similarity scores between query
                molecule and all other molecules of the molecule set.

        """
        query_molecule.match_fingerprint_from(self.molecule_database[0])
        set_similarity = [
            query_molecule.get_similarity_to(
                set_molecule, similarity_measure=self.similarity_measure
            )
            for set_molecule in self.molecule_database
        ]
        return np.array(set_similarity)

    def get_most_similar_pairs(self):
        """Get pairs of samples which are most similar.

        Returns:
            List(Tuple(Molecule, Molecule))
                List of pairs of Molecules closest to one another.
                Since ties are broken randomly, this may be non-transitive
                i.e. (A, B) =/=> (B, A)

        Raises:
            NotInitializedError: If MoleculeSet object does not have
                similarity_measure attribute.

        """
        if self.similarity_matrix is None:
            raise NotInitializedError(
                "MoleculeSet instance not properly "
                "initialized with descriptor and "
                "similarity measure"
            )
        out_list = []
        n_samples = self.similarity_matrix.shape[0]
        for index, row in enumerate(self.similarity_matrix):
            post_diag_closest_index = (
                np.argmax(row[(index + 1):]) + index + 1
                if index < n_samples - 1
                else -1
            )
            pre_diag_closest_index = np.argmax(row[:index]) if index > 0 else -1
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
                    post_diag_closest_index
                    if row[post_diag_closest_index] >= row[pre_diag_closest_index]
                    else pre_diag_closest_index
                )
            out_list.append(
                (self.molecule_database[index], self.molecule_database[closest_index])
            )
        return out_list

    def get_most_dissimilar_pairs(self):
        """Get pairs of samples which are least similar.

        Returns:
            List(Tuple(Molecule, Molecule))
                List of pairs of indices closest to one another.
        Raises:
            NotInitializedError: If MoleculeSet object does not have
                similarity_measure attribute.

        """
        if self.similarity_matrix is None:
            raise NotInitializedError(
                "MoleculeSet instance not properly "
                "initialized with descriptor and "
                "similarity measure"
            )

        out_list = []
        for index, row in enumerate(self.similarity_matrix):
            furthest_index = np.argmin(row)
            out_list.append(
                (self.molecule_database[index], self.molecule_database[furthest_index])
            )
        return out_list

    def get_property_of_most_similar(self):
        """Get property of pairs of molecules
        which are most similar to each other.

        Returns:
            (tuple): The first index is an array of reference mol
            properties and the second index is an array of the
            property of the respective most similar molecule. Skips pairs
            of molecules for which molecule properties are not initialized.

        """
        similar_mol_pairs = self.get_most_similar_pairs()
        reference_mol_properties, similar_mol_properties = [], []
        for mol_pair in similar_mol_pairs:
            mol1_property = mol_pair[0].get_mol_property_val()
            mol2_property = mol_pair[1].get_mol_property_val()
            if mol1_property is not None and mol2_property is not None:
                reference_mol_properties.append(mol1_property)
                similar_mol_properties.append(mol2_property)
        return reference_mol_properties, similar_mol_properties

    def get_property_of_most_dissimilar(self):
        """Get property of pairs of molecule
        which are most dissimilar to each other.

        Returns:
            (tuple): The first index is an array of reference mol
            properties and the second index is an array of the
            property of the respective most dissimilar molecule. Skips pairs
            of molecules for which molecule properties are not initialized.

        """
        dissimilar_mol_pairs = self.get_most_dissimilar_pairs()
        reference_mol_properties, dissimilar_mol_properties = [], []
        for mol_pair in dissimilar_mol_pairs:
            mol1_property = mol_pair[0].get_mol_property_val()
            mol2_property = mol_pair[1].get_mol_property_val()
            if mol1_property is not None and mol2_property is not None:
                reference_mol_properties.append(mol1_property)
                dissimilar_mol_properties.append(mol2_property)
        return reference_mol_properties, dissimilar_mol_properties

    def get_similarity_matrix(self):
        """Get the similarity matrix for the data set.

        Returns:
            (np.ndarray): Similarity matrix of the dataset.
                Shape (n_samples, n_samples).

        Note:
            If un-set, sets the self.similarity_matrix attribute.

        """
        if self.similarity_matrix is None:
            self._set_similarity_matrix()
        return self.similarity_matrix

    def get_distance_matrix(self):
        """Get the distance matrix for the data set.
        This is can only be done for similarity measures which yields
        valid distances.

        Returns:
            (np.ndarray): Distance matrix of the dataset.
                Shape (n_samples, n_samples).

        """
        if not hasattr(self.similarity_measure, "to_distance"):
            raise InvalidConfigurationError(
                f"{self.similarity_measure.metric} "
                f"does not have an equivalent "
                f"distance"
            )
        return self.similarity_measure.to_distance(self.similarity_matrix)

    def get_pairwise_similarities(self):
        """Get an array of pairwise similarities of molecules in the set.

        Returns:
            (np.ndarray): Array of pairwise similarities of the molecules in
            the set. Self similarities are not calculated.

        """
        pairwise_similarity_vector = []
        for ref_mol in range(len(self.molecule_database)):
            for target_mol in range(ref_mol + 1, len(self.molecule_database)):
                pairwise_similarity_vector.append(
                    self.similarity_matrix[ref_mol, target_mol]
                )
        return np.array(pairwise_similarity_vector)

    def get_mol_names(self):
        """Get names of the molecules in the set. This is the Molecule.mol_text
        attribute of the Molecule objects in the MoleculeSet. If this attribute
        is not present, then collection of mol_ids in the form
        "id: " + str(mol_id) is returned.

        Returns:
            np.ndarray: Array with molecules names.

        """
        mol_names = []
        for mol_id, mol in enumerate(self.molecule_database):
            mol_name = mol.get_name()
            if mol_name is None:
                mol_names.append("id: " + str(mol_id))
            else:
                mol_names.append(mol_name)
        return np.array(mol_names)

    def get_mol_properties(self):
        """Get properties of all the molecules in the dataset.
           If all molecules don't have properties, None is returned.
        Returns:
           np.ndarray or None: Array with molecules properties or None.

        """
        mol_properties = []
        for mol in self.molecule_database:
            mol_property = mol.get_mol_property_val()
            if mol_property is None:
                return None
            mol_properties.append(mol_property)
        return np.array(mol_properties)

    def get_mol_features(self):
        """Get features of the molecules in the set.

        Returns:
            np.ndarray: (n_molecules, feature_dimensionality) array.

        """
        mol_features = [mol.get_descriptor_val() for mol in self.molecule_database]
        return np.array(mol_features)

    def cluster(self, n_clusters=8, clustering_method=None, **kwargs):
        """Cluster the molecules of the MoleculeSet.

        Args:
            n_clusters (int): Number of clusters. Default is 8.
            clustering_method (str): Clustering algorithm to use. Default is
                None in which case the algorithm is chosen from the
                similarity measure in use. Implemented clustering_methods are:
                'kmedoids': for the K-Medoids algorithm [1].
                    This method is useful
                    when the molecular descriptors are continuous / Euclidean
                    since it relies on the existence of a sensible medoid.
                'complete_linkage', 'complete':
                    Complete linkage agglomerative hierarchical clustering [2].
                'average_linkage', 'average':
                    average linkage agglomerative hierarchical clustering [2].
                'single_linkage', 'single':
                    single linkage agglomerative hierarchical clustering [2].
                'ward':
                    for Ward's algorithm [2]. This method is useful for
                    Euclidean descriptors.
            kwargs (keyword args): Key word arguments to supply to clustering
                algorithm. See the documentation pages
                listed below for these arguments:
                'kmedoids': https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
                'complete_linkage', 'average_linkage', 'single_linkage', 'ward'
                    : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

        Returns:
            cluster_grouped_mol_names (dict): Dictionary of cluster id
                (key) --> Names of molecules in cluster.

        References:
        [1] Hastie, T., Tibshirani R. and Friedman J.,
            The Elements of statistical Learning: Data Mining, Inference,
            and Prediction, 2nd Ed., Springer Series in Statistics (2009).
        [2] Murtagh, F. and Contreras, P., Algorithms for hierarchical
            clustering: an overview. WIREs Data Mining Knowl Discov
            (2011). https://doi.org/10.1002/widm.53

        """
        if not self.similarity_measure.is_distance_metric():
            raise InvalidConfigurationError(
                str(self.similarity_measure) + " is not a distance metric. "
                "Clustering will not yield "
                "meaningful results."
            )
        if (
            clustering_method == "kmedoids" or clustering_method == "ward"
        ) and self.similarity_measure.type_ == "discrete":
            print(
                f"{clustering_method} cannot be used with "
                f"{self.similarity_measure.type_} "
                f"similarity measure. Changing."
            )
            clustering_method = None
        if clustering_method is None:
            if self.similarity_measure.type_ == "continuous":
                clustering_method = "kmedoids"
            else:
                clustering_method = "complete_linkage"
        self.clusters_ = Cluster(
            n_clusters=n_clusters, clustering_method=clustering_method, **kwargs
        ).fit(self.get_distance_matrix())

    def get_cluster_labels(self):
        """
        Get cluster membership of Molecules.
        Raises:
            NotInitializedError: If MoleculeSet object not clustered.

        """
        try:
            return self.clusters_.get_labels()
        except AttributeError as e:
            raise NotInitializedError(
                "Molecule set not clustered. " "Use cluster() to cluster."
            )

    def get_transformed_descriptors(self, method_="pca", **kwargs):
        """Use an embedding method to transform molecular descriptor to a
        low dimensional representation.

        Args:
            method_ (str): The method used for generating lower dimensional
                embedding. Implemented methods are:
                'pca': Principal Component Analysis [1]
                'mds': Multidimensional scaling [2-4]
                'tsne': t-SNE [5]
                'isomap': Isomap [6]
                'spectral_embedding': Spectral Embedding [7]
            kwargs (dict): Keyword arguments to modify the behaviour of
                the respective embedding methods. See the documentation pages
                listed below for these arguments.
                'pca': https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
                'mds': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
                'tsne': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
                'isomap': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
                'spectral_embedding': https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
        Returns:
            X (np.ndarray): Transformed descriptors of shape
                (n_samples, n_components).
        Raises:
            InvalidConfigurationError: If illegal method_ passed.

        References:
            [1] Bishop, C. M., Pattern recognition and machine learning. 2006.
            [2] Borg, I. and P. Groenen, Modern Multidimensional Scaling:
                Theory and Applications (Springer Series in Statistics). 2005.
            [3] Kruskal, J., Nonmetric multidimensional scaling:
                A numerical method. Psychometrika, 1964. 29(2): p. 115-129.
            [4]	Kruskal, J., Multidimensional scaling by optimizing goodness
                of fit to a nonmetric hypothesis. Psychometrika, 1964.
                29: p. 1-27.
            [5] van der Maaten, L. and G. Hinton, Viualizing data using t-SNE.
                Journal of Machine Learning Research, 2008. 9: p. 2579-2605.
            [6] Tenenbaum, J.B., V.d. Silva, and J.C. Langford,
                A Global Geometric Framework for Nonlinear Dimensionality
                Reduction. Science, 2000. 290(5500): p. 2319-2323.
            [7] Ng, A.Y., M.I. Jordan, and Y. Weiss. On Spectral Clustering:
                Analysis and an algorithm. 2001. MIT Press.

        """
        if method_.lower() == "pca":
            return self._do_pca(**kwargs)
        elif method_.lower() == "mds":
            return self._do_mds(**kwargs)
        elif method_.lower() == "tsne":
            return self._do_tsne(**kwargs)
        elif method_.lower() == "isomap":
            return self._do_isomap(**kwargs)
        elif method_.lower() == "spectral_embedding":
            return self._do_spectral_embedding(**kwargs)
        else:
            raise InvalidConfigurationError(
                f"Embedding method {method_} " f"not implemented"
            )
