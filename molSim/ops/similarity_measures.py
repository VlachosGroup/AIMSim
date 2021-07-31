"""This module contains methods to find similarities between molecules."""
import numpy as np
from rdkit import DataStructs
from scipy.spatial.distance import cosine as scipy_cosine

SMALL_NUMBER = 1e-10


class SimilarityMeasure:
    def __init__(self, metric):
        if metric.lower() in ['negative_l0']:
            self.metric = 'negative_l0'
            self.type_ = 'continuous'
            self.to_distance = lambda x: -x

        elif metric.lower() in ['negative_l1', 'negative_manhattan']:
            self.metric = 'negative_l1'
            self.type_ = 'continuous'
            self.to_distance = lambda x: -x

        elif metric.lower() in ['negative_l2', 'negative_euclidean']:
            self.metric = 'negative_l2'
            self.type_ = 'continuous'
            self.to_distance = lambda x: -x

        elif metric.lower() in ['cosine', 'driver-kroeber', 'ochiai']:
            self.metric = 'cosine'
            self.type_ = 'continuous'
            # angular distance
            self.to_distance = lambda x: np.arccos(x) / np.pi

        elif metric.lower() in ['dice', 'sorenson', 'gleason']:
            self.metric = 'dice'
            self.type_ = 'discrete'
            # convert to jaccard for distance
            self.to_distance = lambda x: 1 - x / (2 - x)

        elif metric.lower() in ['tanimoto', 'jaccard-tanimoto']:
            self.metric = 'tanimoto'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['simple_matching', 'sokal-michener', 'rand']:
            self.metric = 'simple_matching'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['rogers-tanimoto']:
            self.metric = 'rogers_tanimoto'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['russel-rao']:
            self.metric = 'russel_rao'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['forbes']:
            self.metric = 'forbes'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['simpson']:
            self.metric = 'simpson'
            self.type_ = 'discrete'

        elif metric.lower() in ['braun-blanquet']:
            self.metric = 'braun_blanquet'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['baroni-urbani-buser']:
            self.metric = 'baroni_urbani_buser'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['kulczynski']:
            self.metric = 'kulczynski'
            self.type_ = 'discrete'

        elif metric.lower() in ['sokal-sneath', 'sokal-sneath_1']:
            self.metric = 'sokal_sneath'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['sokal-sneath_2',
                                'sokal-sneath-2',
                                'symmetric_sokal_sneath',
                                'symmetric-sokal-sneath']:
            self.metric = 'symmetric_sokal_sneath'
            self.type_ = 'discrete'

        elif metric.lower() in ['sokal-sneath-3', 'sokal-sneath_3']:
            self.metric = 'sokal_sneath_3'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['sokal-sneath-4', 'sokal-sneath_4']:
            self.metric = 'sokal_sneath_4'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['jaccard']:
            self.metric = 'jaccard'
            self.type_ = 'discrete'

        elif metric.lower() in ['faith']:
            self.metric = 'faith'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['michael']:
            self.metric = 'michael'
            self.type_ = 'discrete'

        elif metric.lower() in ['mountford']:
            self.metric = 'mountford'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['rogot-goldberg']:
            self.metric = 'rogot_goldberg'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['hawkins-dotson']:
            self.metric = 'hawkins_dotson'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['maxwell-pilliner']:
            self.metric = 'maxwell_pilliner'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['harris-lahey']:
            self.metric = 'harris_lahey'
            self.type_ = 'discrete'

        elif metric.lower() in ['consonni−todeschini-1',
                                'consonni−todeschini_1']:
            self.metric = 'consonni_todeschini_1'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['consonni−todeschini-2',
                                'consonni−todeschini_2']:
            self.metric = 'consonni_todeschini_2'
            self.type_ = 'discrete'

        elif metric.lower() in ['consonni−todeschini-3',
                                'consonni−todeschini_3']:
            self.metric = 'consonni_todeschini_3'
            self.type_ = 'discrete'

        elif metric.lower() in ['consonni−todeschini-4',
                                'consonni−todeschini_4']:
            self.metric = 'consonni_todeschini_4'
            self.type_ = 'discrete'

        elif metric.lower() in ['consonni−todeschini-5',
                                'consonni−todeschini_5']:
            self.metric = 'consonni_todeschini_5'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['austin-colwell']:
            self.metric = 'austin_colwell'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['yule-1', 'yule_1']:
            self.metric = 'yule_1'
            self.type_ = 'discrete'

        elif metric.lower() in ['yule-2', 'yule_2']:
            self.metric = 'yule_2'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['fossum', 'holiday-fossum', 'holiday_fossum']:
            self.metric = 'fossum'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['dennis', 'holiday-dennis', 'holiday_dennis']:
            self.metric = 'dennis'
            self.type_ = 'discrete'
            self.to_distance = lambda x: 1 - x

        elif metric.lower() in ['cole-1', 'cole_1']:
            self.metric = 'cole_1'
            self.type_ = 'discrete'

        elif metric.lower() in ['cole-2', 'cole_2']:
            self.metric = 'cole_2'
            self.type_ = 'discrete'

        elif metric.lower() in ['dispersion', 'choi']:
            self.metric = 'dispersion'
            self.type_ = 'discrete'

        elif metric.lower() in ['goodman−kruskal']:
            self.metric = 'goodman_kruskal'
            self.type_ = 'discrete'

        else:
            raise ValueError(f"Similarity metric: {metric} is not implemented")
        self.normalize_fn = {'shift_': 0., 'scale_': 1.}
        self.label_ = metric

    def __call__(self, mol1_descriptor, mol2_descriptor):
        """Compare two descriptors.

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            similarity_ (float): Similarity value
        """
        similarity_ = None
        if self.metric == 'negative_l0':
            similarity_ = -np.linalg.norm(
                mol1_descriptor.to_numpy() - mol2_descriptor.to_numpy(), ord=0
            )

        elif self.metric == 'negative_l1':
            similarity_ = -np.linalg.norm(
                mol1_descriptor.to_numpy() - mol2_descriptor.to_numpy(), ord=1
            )

        elif self.metric == 'negative_l2':
            similarity_ = -np.linalg.norm(
                mol1_descriptor.to_numpy() - mol2_descriptor.to_numpy(), ord=2
            )

        elif self.metric == 'austin_colwell':
            try:
                similarity_ = self._get_austin_colwell(mol1_descriptor,
                                                       mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'baroni_urbani_buser':
            try:
                similarity_ = self._get_baroni_urbani_buser(mol1_descriptor,
                                                            mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'braun_blanquet':
            try:
                similarity_ = self._get_braun_blanquet(mol1_descriptor,
                                                       mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'cosine':
            if mol1_descriptor.rdkit_ and mol2_descriptor.rdkit_:
                similarity_ = DataStructs.CosineSimilarity(
                    mol1_descriptor.rdkit_, mol2_descriptor.rdkit_
                )
            else:
                similarity_ = scipy_cosine(
                    mol1_descriptor.to_numpy(), mol2_descriptor.to_numpy()
                )

        elif self.metric == 'cole_1':
            try:
                similarity_ = self._get_cole_1(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'cole_2':
            try:
                similarity_ = self._get_cole_2(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'consonni_todeschini_1':
            try:
                similarity_ = self._get_consonni_todeschini_1(mol1_descriptor,
                                                              mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'consonni_todeschini_2':
            try:
                similarity_ = self._get_consonni_todeschini_2(mol1_descriptor,
                                                              mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'consonni_todeschini_3':
            try:
                similarity_ = self._get_consonni_todeschini_3(mol1_descriptor,
                                                              mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'consonni_todeschini_4':
            try:
                similarity_ = self._get_consonni_todeschini_4(mol1_descriptor,
                                                              mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'consonni_todeschini_5':
            try:
                similarity_ = self._get_consonni_todeschini_5(mol1_descriptor,
                                                              mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'dennis':
            try:
                similarity_ = self._get_dennis(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'dice':
            try:
                similarity_ = DataStructs.DiceSimilarity(
                    mol1_descriptor.to_rdkit(), mol2_descriptor.to_rdkit()
                )
            except ValueError as e:
                raise ValueError(
                    "Dice similarity is only useful for bit strings "
                    "generated from fingerprints. Consider using "
                    "other similarity measures for arbitrary vectors."
                )

        elif self.metric == 'dispersion':
            try:
                similarity_ = self._get_dispersion(mol1_descriptor,
                                                   mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'faith':
            try:
                similarity_ = self._get_faith(mol1_descriptor,
                                              mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'forbes':
            try:
                similarity_ = self._get_forbes(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'fossum':
            try:
                similarity_ = self._get_fossum(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'goodman_kruskal':
            try:
                similarity_ = self._get_goodman_kruskal(mol1_descriptor,
                                                        mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'harris_lahey':
            try:
                similarity_ = self._get_harris_lahey(mol1_descriptor,
                                                     mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'hawkins_dotson':
            try:
                similarity_ = self._get_hawkins_dotson(mol1_descriptor,
                                                       mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'jaccard':
            try:
                similarity_ = self._get_jaccard(mol1_descriptor,
                                                mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'kulczynski':
            try:
                similarity_ = self._get_kulczynski(mol1_descriptor,
                                                   mol2_descriptor)
            except ValueError as e:
                raise e
        elif self.metric == 'maxwell_pilliner':
            try:
                similarity_ = self._get_maxwell_pilliner(mol1_descriptor,
                                                         mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'michael':
            try:
                similarity_ = self._get_michael(mol1_descriptor,
                                                mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'mountford':
            try:
                similarity_ = self._get_mountford(mol1_descriptor,
                                                  mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'rogers_tanimoto':
            try:
                similarity_ = self._get_rogers_tanimoto(mol1_descriptor,
                                                        mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'rogot_goldberg':
            try:
                similarity_ = self._get_rogot_goldberg(mol1_descriptor,
                                                       mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'russel_rao':
            try:
                similarity_ = self._get_russel_rao(mol1_descriptor,
                                                   mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'simple_matching':
            try:
                similarity_ = self._get_simple_matching(mol1_descriptor,
                                                        mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'simpson':
            try:
                similarity_ = self._get_simpson(mol1_descriptor,
                                                mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'sokal_sneath':
            try:
                similarity_ = self._get_sokal_sneath(mol1_descriptor,
                                                     mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'symmetric_sokal_sneath':
            try:
                similarity_ = self._get_symmetric_sokal_sneath(mol1_descriptor,
                                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'sokal_sneath_3':
            try:
                similarity_ = self._get_sokal_sneath_3(mol1_descriptor,
                                                       mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'sokal_sneath_4':
            try:
                similarity_ = self._get_sokal_sneath_4(mol1_descriptor,
                                                       mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'tanimoto':
            try:
                similarity_ = DataStructs.TanimotoSimilarity(
                    mol1_descriptor.to_rdkit(), mol2_descriptor.to_rdkit()
                )
            except ValueError as e:
                raise e
            except ValueError as e:
                raise ValueError(
                    "Tanimoto similarity is only useful for bit strings "
                    "generated from fingerprints. Consider using "
                    "other similarity measures for arbitrary vectors."
                )
        elif self.metric == 'yule_1':
            try:
                similarity_ = self._get_yule_1(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        elif self.metric == 'yule_2':
            try:
                similarity_ = self._get_yule_2(mol1_descriptor,
                                               mol2_descriptor)
            except ValueError as e:
                raise e

        else:
            raise ValueError(f'{self.metric} could not be implemented')

        return similarity_

    def _get_austin_colwell(self, mol1_descriptor, mol2_descriptor):
        """Calculate Austin-Colwell similarity between two molecules.
        This is defined for two binary arrays as:
        Austin-Colwell similarity = (2 / pi) * arcsin(sqrt( (a + d) / p ))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Austin-Colwell similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Austin-Colwell similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (2 / np.pi) * np.arcsin(np.sqrt( (a + d) / p ))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_baroni_urbani_buser(self, mol1_descriptor, mol2_descriptor):
        """Calculate Baroni-Urbani-Buser similarity between two molecules.
        This is defined for two binary arrays as:
        Baroni-Urbani-Buser similarity = (sqrt(a * d) + a)
                                         / (sqrt(a * d) + a + b + c)
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Baroni-Urbani-Buser similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Baroni-Urbani-Buser similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if d == p:
            return 1.
        similarity_ = (np.sqrt(a * d)
                       + a) / (np.sqrt(a * d) + a + b + c)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_braun_blanquet(self, mol1_descriptor, mol2_descriptor):
        """Calculate braun-blanquet similarity between two molecules.
        This is defined for two binary arrays as:
        Braun-Blanquet Similarity = a / max{(a + b), (a + c)}

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Braun-Blanquet similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Braun-Blanquet similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if a < SMALL_NUMBER:
            return 0.
        similarity_ = a / max((a + b), (a + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_cole_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Cole(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Cole(1) Similarity = (a*d - b*c) / ((a + c)*(c + d))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Cole(1)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Cole(1) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if (a + c) < SMALL_NUMBER and (c + d) < SMALL_NUMBER:
            return 0.
        similarity_ = (a*d - b*c) / ((a + c)*(c + d))
        self.normalize_fn["shift_"] = p - 1
        self.normalize_fn["scale_"] = p
        return self._normalize(similarity_)

    def _get_cole_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Cole(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Cole(2) Similarity = (a*d - b*c) / ((a + b)*(b + d))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Cole(2)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Cole(2) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if (a + b) < SMALL_NUMBER and (b + d) < SMALL_NUMBER:
            return 0.
        similarity_ = (a*d - b*c) / ((a + b)*(b + d))
        self.normalize_fn["shift_"] = p - 1
        self.normalize_fn["scale_"] = p
        return self._normalize(similarity_)

    def _get_consonni_todeschini_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(1) similarity = ln(1 + a + d) / ln(1 + p)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(1)  similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(1)  similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = np.log(1 + a + d) / np.log(1 + p)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_consonni_todeschini_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(2) similarity =
            (ln(1 + p) - ln(1 + b + c)) / ln(1 + p)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(2)  similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(2) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (np.log(1 + p) - np.log(1 + b + c)) / np.log(1 + p)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_consonni_todeschini_3(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(3) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(3) similarity = ln(1 + a) / ln(1 + p)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(3)  similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(3) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = np.log(1 + a) / np.log(1 + p)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_consonni_todeschini_4(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(4) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(4) similarity = ln(1 + a) / ln(1 + a + b + c)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(4)  similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(4) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        similarity_ = np.log(1 + a) / np.log(1 + a + b + c)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_consonni_todeschini_5(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(5) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(5) similarity = (ln(1 + a*d) - ln(1 + b*c))
                                            / ln(1 + p**2 / 4)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(5)  similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(5) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (np.log(1 + a*d) - np.log(1 + b*c)) / np.log(1 + p**2 / 4)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_dennis(self, mol1_descriptor, mol2_descriptor):
        """Calculate Dennis similarity between two molecules.
        This is defined for two binary arrays as:
        Dennis similarity = (a*d - b*c) / sqrt(p*(a + b)*(a + c))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Dennis similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dennis similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if (a + b) < SMALL_NUMBER and (b + c) < SMALL_NUMBER:
            return 0.
        similarity_ = (a*d - b*c) / np.sqrt(p*(a + b)*(a + c))
        self.normalize_fn["shift_"] = np.sqrt(p) / 2
        self.normalize_fn["scale_"] = np.sqrt(p)
        return self._normalize(similarity_)

    def _get_dispersion(self, mol1_descriptor, mol2_descriptor):
        """Calculate dispersion similarity in Choi et al (2012)
         between two molecules. This is defined for two binary arrays as:
        dispersion similarity = (a*d - b*c) / p**2

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): dispersion similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dispersion similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        similarity_ = (a*d - b*c) / p**2
        self.normalize_fn["shift_"] = 1 / 4
        self.normalize_fn["scale_"] = 1 / 2
        return self._normalize(similarity_)

    def _get_faith(self, mol1_descriptor, mol2_descriptor):
        """Calculate faith similarity between two molecules.
        This is defined for two binary arrays as:
        Faith similarity = (a + 0.5*d) / p
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Faith similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Faith similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (a + 0.5 * d) / p
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_forbes(self, mol1_descriptor, mol2_descriptor):
        """Calculate forbes similarity between two molecules.
        This is defined for two binary arrays as:
        Forbes similarity = (p * a) / ((a + b) * (a + c))
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Forbes similarity value
        
        Note:
            The Forbes similarity is normalized to [0, 1]
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Forbes similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if (a + b) < SMALL_NUMBER or (a + c) < SMALL_NUMBER or a < SMALL_NUMBER:
            return 0.
        p = a + b + c + d
        similarity_ = (p * a) / ((a + b) * (a + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = p / a
        return self._normalize(similarity_)

    def _get_fossum(self, mol1_descriptor, mol2_descriptor):
        """Calculate Fossum similarity between two molecules.
        This is defined for two binary arrays as:
        Fossum similarity = p * (a - 0.5)**2 / ((a + b) * (a + c))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Fossum similarity value

        Note:
            The similarity is normalized to [0, 1].
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Fossum similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if (a + b) < SMALL_NUMBER or (a + c) < SMALL_NUMBER:
            return 0.
        p = a + b + c + d
        similarity_ = p * (a - 0.5)**2 / ((a + b) * (a + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = (p - 0.5)**2 / p
        return self._normalize(similarity_)

    def _get_goodman_kruskal(self, mol1_descriptor, mol2_descriptor):
        """Calculate Goodman-Kruskal similarity between two molecules.
        This is defined for two binary arrays as:
        Goodman-Kruskal similarity =
            (2 * min(a, d) - b - c) / (2 * min(a, d) + b + c)
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Goodman-Kruskal similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Goodman-Kruskal similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        min_a_d = np.min([a, d])
        similarity_ = (2 * min_a_d - b - c) / (2 * min_a_d + b + c)
        self.normalize_fn["shift_"] = 1.
        self.normalize_fn["scale_"] = 2.
        return self._normalize(similarity_)

    def _get_harris_lahey(self, mol1_descriptor, mol2_descriptor):
        """Calculate Harris-Lahey similarity between two molecules.
        This is defined for two binary arrays as:
        Harris-Lahey similarity =
            (a/2) * (2*d + b + c)/(a + b + c)
             + (d/2) * (2*a + b + c)/( b + c + d)
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Harris-Lahey similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Harris-Lahey similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if (a + b + c) < SMALL_NUMBER or (b + c + d) < SMALL_NUMBER:
            return 0.

        similarity_ = ((a / 2) * (2 * d + b + c) / (a + b + c)
                       + (d / 2) * (2 * a + b + c) / (b + c + d))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = p
        return self._normalize(similarity_)

    def _get_hawkins_dotson(self, mol1_descriptor, mol2_descriptor):
        """Calculate Hawkins-Dotson similarity between two molecules.
        This is defined for two binary arrays as:
        Hawkins-Dotson similarity = 0.5 * (a / (a + b + c)
                                           + d / (d + b + c))
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Hawkins-Dotson similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Hawkins-Dotson similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        similarity_ = 0.5 * (a / (a + b + c) + d / (d + b + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_jaccard(self, mol1_descriptor, mol2_descriptor):
        """Calculate jaccard similarity between two molecules.
        This is defined for two binary arrays as:
        jaccard similarity = 3*a / (3*a + b + c)
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Jaccard similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Jaccard similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if a == 0:
            return 0.
        similarity_ = 3 * a / (3 * a + b + c)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_kulczynski(self, mol1_descriptor, mol2_descriptor):
        """Calculate kulczynski similarity between two molecules.
        This is defined for two binary arrays as:
        kulczynski similarity = 0.5 * a / ((a + b) + (a + c))
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Kulczynski similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Kulczynski similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if a == 0:
            return 0.
        similarity_ = 0.5 * a / ((a + b) + (a + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_michael(self, mol1_descriptor, mol2_descriptor):
        """Calculate michael similarity between two molecules.
        This is defined for two binary arrays as:
        michael similarity = 4*(a*d - b*c) / ((a + d)**2 + (b + c)**2)
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Michael similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Michael similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p or (b + c) == 0:
            return 1.
        similarity_ = 4 * (a * d - b * c) / ((a + d) ** 2 + (b + c) ** 2)
        self.normalize_fn["shift_"] = 1.
        self.normalize_fn["scale_"] = 2.
        return self._normalize(similarity_)

    def _get_maxwell_pilliner(self, mol1_descriptor, mol2_descriptor):
        """Calculate Maxwell-Pilliner similarity between two molecules.
                This is defined for two binary arrays as:
                Maxwell-Pilliner similarity =
                    2*(a*d - b*c) / ((a + b)*(c + d) + (a + c)*(b + d))

                Args:
                    mol1_descriptor (molSim.ops Descriptor)
                    mol2_descriptor (molSim.ops Descriptor)

                Returns:
                    (float): Maxwell-Pilliner similarity value
                """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Maxwell-Pilliner similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if (a + b) * (c + d) + (a + c) * (b + d) < SMALL_NUMBER:
            return 0.

        similarity_ = 2 * (a * d - b * c) / (
                    (a + b) * (c + d) + (a + c) * (b + d))
        self.normalize_fn["shift_"] = 1.
        self.normalize_fn["scale_"] = 2.
        return self._normalize(similarity_)

    def _get_mountford(self, mol1_descriptor, mol2_descriptor):
        """Calculate mountford similarity between two molecules.
        This is defined for two binary arrays as:
        mountford similarity = 2*a / (a*b + a*c + 2*b*c)
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Mountford similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Mountford similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a * b + a * c + 2 * b * c == 0:
            return a / p
        similarity_ = 2 * a / (a * b + a * c + 2 * b * c)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 2.
        return self._normalize(similarity_)

    def _get_rogers_tanimoto(self, mol1_descriptor, mol2_descriptor):
        """Calculate rogers-tanimoto similarity between two molecules.
        This is defined for two binary arrays as:
        Rogers-Tanimoto similarity = (a + d) / (p + b + c)
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Rogers-Tanimoto similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Rogers-Tanimoto similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (a + d) / (p + b + c)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_rogot_goldberg(self, mol1_descriptor, mol2_descriptor):
        """Calculate Rogot-Goldberg similarity between two molecules.
        This is defined for two binary arrays as:
        Rogot-Goldberg similarity = (a / (2*a + b + c)) + (d / (2*d + b + c))
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Rogot-Goldberg  similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Rogot-Goldberg  similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        similarity_ = (a / (2 * a + b + c)) + (d / (2 * d + b + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_russel_rao(self, mol1_descriptor, mol2_descriptor):
        """Calculate russel-rao similarity between two molecules.
        This is defined for two binary arrays as:
        Russel-Rao = a / p
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Russel-Rao similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Russel-Rao similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = a / p
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_simple_matching(self, mol1_descriptor, mol2_descriptor):
        """Calculate simple matching similarity between two molecules.
        This is defined for two binary arrays as:
        Simple Matching = (a + d) / p
        
        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Simple Matching similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Simple Matching similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (a + d) / p
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_simpson(self, mol1_descriptor, mol2_descriptor):
        """Calculate simpson similarity between two molecules.
        This is defined for two binary arrays as:
        Simpson Similarity = a / min{(a + b), (a + c)}

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Simpson similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Simpson similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if (a + b) == 0 or (a + c) == 0 or a == 0:
            return 0.
        similarity_ = a / min((a + b), (a + c))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_sokal_sneath(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sokal-Sneath similarity between two molecules.
        This is defined for two binary arrays as:
        Sokal-Sneath similarity = a / (a + 2*b + 2*c)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Sokal-Sneath similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sokal-Sneath similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        if a == 0:
            return 0.
        similarity_ = a / (a + 2 * b + 2 * c)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_symmetric_sokal_sneath(self, mol1_descriptor, mol2_descriptor):
        """Calculate Symmetric Sokal-Sneath similarity between two molecules.
        This is defined for two binary arrays as:
        Symmetric Sokal-Sneath similarity = (2*a + 2*d) / (p + a + d)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Symmetric Sokal-Sneath similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Symmetric Sokal-Sneath similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        similarity_ = (2 * a + 2 * d) / (p + a + d)
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_sokal_sneath_3(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sokal-Sneath(3) similarity between two molecules.
        This is defined for two binary arrays as:
        Sokal-Sneath(3) similarity =
        (1/4) * (a/(a + b) + a/(a + c) + d/(b + d) + d/(c + d))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Sokal-Sneath(3) similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sokal-Sneath(3) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if a < SMALL_NUMBER and d < SMALL_NUMBER:
            return 0.
        similarity_ = (1/4) * (a/(a + b) + a/(a + c) + d/(b + d) + d/(c + d))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_sokal_sneath_4(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sokal-Sneath(4) similarity between two molecules.
        This is defined for two binary arrays as:
        Sokal-Sneath(4) similarity =
         a/sqrt((a + b) * (a + c)) * d/sqrt((b + d) *(c + d))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Sokal-Sneath(4) similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sokal-Sneath(4) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p:
            return 1.
        if a < SMALL_NUMBER and d < SMALL_NUMBER:
            return 0.
        similarity_ = (a/np.sqrt((a + b) * (a + c))) * (
                                                   d/np.sqrt((b + d) * (c + d)))
        self.normalize_fn["shift_"] = 0.
        self.normalize_fn["scale_"] = 1.
        return self._normalize(similarity_)

    def _get_yule_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Yule(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Symmetric Yule(1) similarity = (a*d - b*c) / (a*d + b*c)

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Yule(1) similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Yule(1) similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p or b*c < SMALL_NUMBER:
            return 1.
        similarity_ = (a*d - b*c) / (a*d + b*c)
        self.normalize_fn["shift_"] = 1.
        self.normalize_fn["scale_"] = 2.
        return self._normalize(similarity_)

    def _get_yule_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Yule(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Symmetric Yule(2) similarity = (sqrt(a*d) - sqrt(b*c))
                                       / (sqrt(a*d) + sqrt(b*c))

        Args:
            mol1_descriptor (molSim.ops Descriptor)
            mol2_descriptor (molSim.ops Descriptor)

        Returns:
            (float): Yule(2) similarity value
        """
        if not (mol1_descriptor.is_fingerprint()
                and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Yule(2) similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor.to_numpy(),
                                    mol2_descriptor.to_numpy())
        p = a + b + c + d
        if a == p or d == p or b*c < SMALL_NUMBER:
            return 1.
        similarity_ = (np.sqrt(a*d)
                       - np.sqrt(b*c)) / (np.sqrt(a*d) + np.sqrt(b*c))
        self.normalize_fn["shift_"] = 1.
        self.normalize_fn["scale_"] = 2.
        return self._normalize(similarity_)

    def _get_abcd(self, arr1, arr2):
        """ Get a, b, c, d, where:
        a = bits(array 1) and bits(array 2)
        b = bits(array 1) and bits(~array 2)
        c = bits(~array 1) and bits(array 2)
        d = bits(~array 1) and bits(~array 2)   // "~": complement operator
        p = a + b + c + d = bits(array 1 or array 2)
        
        Args:
            arr1 (np.ndarray)
            arr2 (np.ndarray)

        Returns:
            (tuple): (a, b, c, d)
        """

        def _to_equal_length(arr1, arr2):
            out_arr = [np.array(arr1), np.array(arr2)]
            max_length = max(arr1.size, arr2.size)
            for arr_id, arr in enumerate(out_arr):
                out_arr[arr_id] = np.pad(arr,
                                         (0, max_length - arr.size),
                                         mode='constant')
            return out_arr

        arr1, arr2 = _to_equal_length(arr1, arr2)
        not_arr1 = np.logical_not(arr1)
        not_arr2 = np.logical_not(arr2)
        a = np.sum(arr1 & arr2)
        b = np.sum(arr1 & not_arr2)
        c = np.sum(not_arr1 & arr2)
        d = np.sum(not_arr1 & not_arr2)
        assert (a + b + c + d) == arr1.size == arr2.size
        return a, b, c, d

    def _normalize(self, similarity_):
        return (similarity_
                + self.normalize_fn['shift_']) / self.normalize_fn['scale_']

    def is_distance_metric(self):
        """Check if the similarity measure is a distance metric.
        
        Returns:
            bool: True if it is a distance metric.
        """
        return hasattr(self, 'to_distance')

    @staticmethod
    def get_supported_metrics():
        """Return a list of strings for the currently implemented
        similarity measures, aka metrics.

        Returns:
            List: List of strings.
        """
        return [
            'negative_l0',
            'negative_l1',
            'negative_manhattan',
            'negative_l2',
            'negative_euclidean',
            'dice',
            'sorenson',
            'gleason',
            'jaccard',
            'tanimoto',
            "cosine",
            'driver-kroeber',
            'ochiai',
            'simple_matching',
            'sokal-michener',
            'rand',
            'rogers-tanimoto',
            'russel-rao',
            'forbes',
            'simpson',
            'braun-blanquet',
            'baroni-urbani-buser',
            'kulczynski',
            'sokal-sneath',
            'sokal-sneath-2',
            'symmetric_sokal_sneath',
            'symmetric-sokal-sneath',
            'sokal-sneath-3',
            'sokal-sneath_3',
            'sokal-sneath-4',
            'sokal-sneath_4',
            'jaccard',
            'faith',
            'mountford',
            'michael',
            'rogot-goldberg',
            'hawkins-dotson',
            'maxwell-pilliner',
            'harris-lahey',
            'consonni−todeschini-1',
            'consonni−todeschini-2',
            'consonni−todeschini-3',
            'consonni−todeschini-4',
            'consonni−todeschini-5',
            'austin-colwell',
            'yule-1',
            'yule_1',
            'yule_2',
            'yule_2',
            'fossum',
            'holiday-fossum',
            'holiday_fossum',
            'dennis',
            'holiday-dennis',
            'holiday_dennis',
            'cole-1',
            'cole_1',
            'cole-2',
            'cole_2',
            'dispersion',
            'choi',
            'goodman−kruskal',
        ]

    def __str__(self):
        return self.label_
