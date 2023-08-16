"""This module contains methods to find similarities between molecules."""
from functools import lru_cache
import numpy as np
from rdkit import DataStructs

from aimsim.ops import Descriptor
from aimsim.exceptions import InvalidConfigurationError

SMALL_NUMBER = 1e-10


# to aggregate all of the different similarity metrics, we use a `register` decorator
# and metaclass to keep track of all of them.
#
# When a new similarity measure is added, it should be decorated with all of the supported
# aliases, the type (if not discrete) and the formula required to convert it to a distance
# metric (i.e. 1 is furthest and o is closest).
#
# The metaclass will then automatically call the method properly when SimilarityMeasure
# is called, as well as export the method name to the global variables imported elsewhere.
registry = {}

ALL_METRICS = []
BINARY_METRICS = []
UNIQUE_METRICS = []
ALIAS_TO_FUNC = {}
ALIAS_TO_DISTANCE = {}
ALIAS_TO_TYPE = {}


def register(*args, type="discrete", to_distance=None):
    def wrapper(func):
        # first arg should be 'preferred' alias
        func._register = (func, type, to_distance, *args)
        return func

    return wrapper


class RegisteringType(type):
    def __init__(cls, name, bases, attrs):
        for key, val in attrs.items():
            registry_data = getattr(val, "_register", None)
            if registry_data is None:
                continue

            # iterate through all the accepted names for each metric
            for alias in registry_data[3:]:
                # save the alias to func mapping
                ALIAS_TO_FUNC[alias] = registry_data[0]
                # save the metric types
                ALIAS_TO_TYPE[alias] = registry_data[1]
                # save the distance conversion functions, where provided
                if registry_data[2] is not None:
                    ALIAS_TO_DISTANCE[alias] = registry_data[2]

            # save only the first metric to the unique metrics list
            UNIQUE_METRICS.append(registry_data[3])

            # add all of the aliases to the metric list
            ALL_METRICS.extend(registry_data[3:])

            # add binary metrics to the appropriate list
            if registry_data[1] == "discrete":
                BINARY_METRICS.extend(registry_data[3:])


class SimilarityMeasure(object, metaclass=RegisteringType):
    def __init__(self, metric):
        lowercase_metric = metric.lower()
        if lowercase_metric not in ALL_METRICS:
            raise ValueError(f"Similarity metric: {metric} is not implemented")

        # check if the chosen metric is a distance metric
        self._is_distance = metric in ALIAS_TO_DISTANCE.keys()
        # assign a function to convert to distance if it is not, otherwise
        # pass through the value
        self.to_distance = ALIAS_TO_DISTANCE.get(metric, lambda x: x)

        # check the registry dictionaries to get the remaining info
        self.metric = lowercase_metric
        self.type_ = ALIAS_TO_TYPE[lowercase_metric]

        self.normalize_fn = {"shift_": 0.0, "scale_": 1.0}
        self.label_ = metric

    def __call__(self, mol1_descriptor, mol2_descriptor):
        """Compare two descriptors.

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            similarity_ (float): Similarity value
        """
        if not self._validate_fprint(mol1_descriptor) or not self._validate_fprint(
            mol2_descriptor
        ):
            raise ValueError(
                f"Molecule descriptor ({mol1_descriptor.label_}) has no active bits."
            )

        value = None
        func = ALIAS_TO_FUNC[self.metric]
        try:
            value = func(self, mol1_descriptor, mol2_descriptor)
        except ValueError as e:
            raise ValueError(
                f"Unexpected error ocurred when calculating {self.metric:s} distance."
                " Original Exception: " + str(e)
            )

        return value

    @register(
        "tanimoto",
        "jaccard-tanimoto",
        to_distance=lambda x: 1 - x,
    )
    def _get_tanimoto(self, mol1_descriptor, mol2_descriptor):
        similarity_ = None
        try:
            similarity_ = DataStructs.TanimotoSimilarity(
                mol1_descriptor.to_rdkit(), mol2_descriptor.to_rdkit()
            )
        except ValueError as e:
            raise ValueError(
                "Tanimoto similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
                " Original Exception: " + str(e)
            )
        return similarity_

    @register("l0_similarity", type="continuous", to_distance=lambda x: 1 - x)
    def _get_l0_similarity(self, mol1_descriptor, mol2_descriptor):
        return self._get_vector_norm_similarity(mol1_descriptor, mol2_descriptor, 0)

    @register(
        "l1_similarity",
        "manhattan_similarity",
        "taxicab_similarity",
        "city_block_similarity",
        "snake_similarity",
        type="continuous",
        to_distance=lambda x: 1 - x,
    )
    def _get_l1_similarity(self, mol1_descriptor, mol2_descriptor):
        return self._get_vector_norm_similarity(mol1_descriptor, mol2_descriptor, 1)

    @register(
        "l2_similarity",
        "euclidean_similarity",
        type="continuous",
        to_distance=lambda x: 1 - x,
    )
    def _get_l2_similarity(self, mol1_descriptor, mol2_descriptor):
        return self._get_vector_norm_similarity(mol1_descriptor, mol2_descriptor, 2)

    def _get_vector_norm_similarity(self, mol1_descriptor, mol2_descriptor, ord):
        """Calculate the norm based similarity between two molecules.
        This is defined as:
        Norm similarity (order n) = 1 / (1 + n-norm(A - B)
        where n-norm(A - B) represents the n-th order norm of the difference
        of two vector A and B.

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)
            ord (int): Order of the norm

        Returns:
            (float): Norm similarity value
        """
        arr1 = mol1_descriptor.to_numpy()
        arr2 = mol2_descriptor.to_numpy()
        if len(arr1) != len(arr2):
            try:
                arr1, arr2 = Descriptor.fold_to_equal_length(
                    mol1_descriptor, mol2_descriptor
                )
            except ValueError as e:
                raise ValueError(
                    "Fingerprints are of unequal length and cannot be folded."
                    " Original Exception: " + str(e)
                )

        norm_ = np.linalg.norm(arr1 - arr2, ord=ord)
        similarity_ = 1 / (1 + norm_)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("austin_colwell", "austin-colwell", to_distance=lambda x: 1 - x)
    def _get_austin_colwell(self, mol1_descriptor, mol2_descriptor):
        """Calculate Austin-Colwell similarity between two molecules.
        This is defined for two binary arrays as:
        Austin-Colwell similarity = (2 / pi) * arcsin(sqrt( (a + d) / p ))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Austin-Colwell similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Austin-Colwell similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (2 / np.pi) * np.arcsin(np.sqrt((a + d) / p))
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("baroni-urbani-buser", to_distance=lambda x: 1 - x)
    def _get_baroni_urbani_buser(self, mol1_descriptor, mol2_descriptor):
        """Calculate Baroni-Urbani-Buser similarity between two molecules.
        This is defined for two binary arrays as:
        Baroni-Urbani-Buser similarity = (sqrt(a * d) + a)
                                         / (sqrt(a * d) + a + b + c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Baroni-Urbani-Buser similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Baroni-Urbani-Buser similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if d == p:
            return 1.0
        similarity_ = (np.sqrt(a * d) + a) / (np.sqrt(a * d) + a + b + c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("braun-blanquet", to_distance=lambda x: 1 - x)
    def _get_braun_blanquet(self, mol1_descriptor, mol2_descriptor):
        """Calculate braun-blanquet similarity between two molecules.
        This is defined for two binary arrays as:
        Braun-Blanquet Similarity = a / max{(a + b), (a + c)}

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Braun-Blanquet similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Braun-Blanquet similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = a / max((a + b), (a + c))
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("cohen")
    def _get_cohen(self, mol1_descriptor, mol2_descriptor):
        """Calculate Cohen similarity between two molecules.
        This is defined for two binary arrays as:
        Cohen Similarity = 2*(a*d - b*c) / ((a + b)*(b + d) + (a + c)*(c + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Cohen similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Cohen similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        denominator_ = (a + b) * (b + d) + (a + c) * (c + d)
        if a == p or d == p:
            return 1.0
        if denominator_ < SMALL_NUMBER:
            return 0.0

        similarity_ = 2 * (a * d - b * c) / denominator_
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("cole_1", "cole-1")
    def _get_cole_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Cole(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Cole(1) Similarity = (a*d - b*c) / ((a + c)*(c + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Cole(1)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Cole(1) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        denominator = (a + c) * (c + d)
        if denominator < SMALL_NUMBER:
            return 0.0
        similarity_ = (a * d - b * c) / denominator
        self.normalize_fn["shift_"] = p - 1
        self.normalize_fn["scale_"] = p
        return self._normalize(similarity_)

    @register("cole_2", "cole-2")
    def _get_cole_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Cole(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Cole(2) Similarity = (a*d - b*c) / ((a + b)*(b + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Cole(2)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Cole(2) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        denominator = (a + b) * (b + d)
        if denominator < SMALL_NUMBER:
            return 0.0
        similarity_ = (a * d - b * c) / denominator
        self.normalize_fn["shift_"] = p - 1
        self.normalize_fn["scale_"] = p
        return self._normalize(similarity_)

    @register(
        "consonni_todeschini_1",
        "consonni-todeschini-1",
        "consonni-todeschini_1",
        to_distance=lambda x: 1 - x,
    )
    def _get_consonni_todeschini_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(1) similarity = ln(1 + a + d) / ln(1 + p)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(1)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(1)  similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = np.log(1 + a + d) / np.log(1 + p)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("consonni_todeschini_2", "consonni-todeschini-2", "consonni-todeschini_2")
    def _get_consonni_todeschini_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(2) similarity =
            (ln(1 + p) - ln(1 + b + c)) / ln(1 + p)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(2)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(2) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (np.log(1 + p) - np.log(1 + b + c)) / np.log(1 + p)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("consonni_todeschini_3", "consonni-todeschini-3", "consonni-todeschini_3")
    def _get_consonni_todeschini_3(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(3) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(3) similarity = ln(1 + a) / ln(1 + p)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(3)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(3) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = np.log(1 + a) / np.log(1 + p)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("consonni_todeschini_4", "consonni-todeschini-4", "consonni-todeschini_4")
    def _get_consonni_todeschini_4(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(4) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(4) similarity = ln(1 + a) / ln(1 + a + b + c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(4)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(4) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if np.log(1 + a + b + c) == 0:
            raise InvalidConfigurationError("Empty string supplied")
        similarity_ = np.log(1 + a) / np.log(1 + a + b + c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register(
        "consonni_todeschini_5",
        "consonni-todeschini-5",
        "consonni-todeschini_5",
        to_distance=lambda x: 1 - x,
    )
    def _get_consonni_todeschini_5(self, mol1_descriptor, mol2_descriptor):
        """Calculate Consonni-Todeschini(5) similarity between two molecules.
        This is defined for two binary arrays as:
        Consonni-Todeschini(5) similarity = (ln(1 + a*d) - ln(1 + b*c))
                                            / ln(1 + p**2 / 4)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Consonni-Todeschini(5)  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Consonni-Todeschini(5) similarity is only useful for "
                "bit strings generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (np.log(1 + a * d) - np.log(1 + b * c)) / np.log(1 + (p**2) / 4)
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register(
        "cosine",
        "driver-kroeber",
        "ochiai",
        to_distance=lambda x: np.arccos(x) / np.pi,
    )
    def _get_cosine_similarity(self, mol1_descriptor, mol2_descriptor):
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        denominator = np.sqrt((a + b) * (a + c))
        if denominator < SMALL_NUMBER:
            return 0.0
        similarity_ = a / denominator
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("dennis", "holiday-dennis", "holiday_dennis", to_distance=lambda x: 1 - x)
    def _get_dennis(self, mol1_descriptor, mol2_descriptor):
        """Calculate Dennis similarity between two molecules.
        This is defined for two binary arrays as:
        Dennis similarity = (a*d - b*c) / sqrt(p*(a + b)*(a + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Dennis similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dennis similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        denominator = np.sqrt(p * (a + b) * (a + c))
        if denominator < SMALL_NUMBER:
            return 0.0
        similarity_ = (a * d - b * c) / denominator
        self.normalize_fn["shift_"] = np.sqrt(p) / 2
        self.normalize_fn["scale_"] = 3 * np.sqrt(p) / 2
        return self._normalize(similarity_)

    @register(
        "dice",
        "sorenson",
        "gleason",
        to_distance=lambda x: 1 - x / (2 - x),
    )
    def _get_dice(self, mol1_descriptor, mol2_descriptor):
        """Calculate Dice similarity between two molecules.
        This is defined for two binary arrays as:
        Dice similarity = 2a / (2a + b + c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Dice similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dice similarity is only useful for bit strings "
                "generated from fingerprints, not {} and {}. Consider using "
                "other similarity measures for arbitrary vectors.".format(
                    mol1_descriptor.get_label(), mol2_descriptor.get_label()
                )
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = 2 * a / (2 * a + b + c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("dice_2")
    def _get_dice_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Dice(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Dice(2) similarity = a / (a + b)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Dice(2) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dice(2) similarity is only useful for bit strings "
                "generated from fingerprints, not {} and {}. Consider using "
                "other similarity measures for arbitrary vectors.".format(
                    mol1_descriptor.get_label(), mol2_descriptor.get_label()
                )
            )
        a, b, _, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = a / (a + b)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("dice_3")
    def _get_dice_3(self, mol1_descriptor, mol2_descriptor):
        """Calculate Dice(3) similarity between two molecules.
        This is defined for two binary arrays as:
        Dice(3) similarity = a / (a + c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Dice(3) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dice(3) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, _, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)

        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = a / (a + c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("dispersion", "choi")
    def _get_dispersion(self, mol1_descriptor, mol2_descriptor):
        """Calculate dispersion similarity in Choi et al (2012)
         between two molecules. This is defined for two binary arrays as:
        dispersion similarity = (a*d - b*c) / p**2

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): dispersion similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Dispersion similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        similarity_ = (a * d - b * c) / p**2
        self.normalize_fn["shift_"] = 1 / 4
        self.normalize_fn["scale_"] = 1 / 2
        return self._normalize(similarity_)

    @register("faith", to_distance=lambda x: 1 - x)
    def _get_faith(self, mol1_descriptor, mol2_descriptor):
        """Calculate faith similarity between two molecules.
        This is defined for two binary arrays as:
        Faith similarity = (a + 0.5*d) / p

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Faith similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Faith similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (a + 0.5 * d) / p
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("forbes", to_distance=lambda x: 1 - x)
    def _get_forbes(self, mol1_descriptor, mol2_descriptor):
        """Calculate forbes similarity between two molecules.
        This is defined for two binary arrays as:
        Forbes similarity = (p * a) / ((a + b) * (a + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Forbes similarity value

        Note:
            The Forbes similarity is normalized to [0, 1]
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Forbes similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if (a + b) * (a + c) < SMALL_NUMBER or a < SMALL_NUMBER:
            return 0.0
        p = a + b + c + d
        similarity_ = (p * a) / ((a + b) * (a + c) + SMALL_NUMBER)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = p / a
        return self._normalize(similarity_)

    @register("fossum", "holiday-fossum", "holiday_fossum", to_distance=lambda x: 1 - x)
    def _get_fossum(self, mol1_descriptor, mol2_descriptor):
        """Calculate Fossum similarity between two molecules.
        This is defined for two binary arrays as:
        Fossum similarity = p * (a - 0.5)**2 / ((a + b) * (a + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Fossum similarity value

        Note:
            The similarity is normalized to [0, 1].
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Fossum similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        denominator = (a + b) * (a + c)
        if denominator < SMALL_NUMBER:
            return 0.0
        p = a + b + c + d
        similarity_ = p * (a - 0.5) ** 2 / denominator
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = (p - 0.5) ** 2 / p
        return self._normalize(similarity_)

    @register("goodman_kruskal", "goodman-kruskal")
    def _get_goodman_kruskal(self, mol1_descriptor, mol2_descriptor):
        """Calculate Goodman-Kruskal similarity between two molecules.
        This is defined for two binary arrays as:
        Goodman-Kruskal similarity =
            (2 * min(a, d) - b - c) / (2 * min(a, d) + b + c)
        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Goodman-Kruskal similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Goodman-Kruskal similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        min_a_d = np.min([a, d])
        similarity_ = (2 * min_a_d - b - c) / (2 * min_a_d + b + c)
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("harris_lahey")
    def _get_harris_lahey(self, mol1_descriptor, mol2_descriptor):
        """Calculate Harris-Lahey similarity between two molecules.
        This is defined for two binary arrays as:
        Harris-Lahey similarity =
            (a/2) * (2*d + b + c)/(a + b + c)
             + (d/2) * (2*a + b + c)/( b + c + d)
        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Harris-Lahey similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Harris-Lahey similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)

        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        if (a + b + c) < SMALL_NUMBER or (b + c + d) < SMALL_NUMBER:
            return 0.0

        similarity_ = (a / 2) * (2 * d + b + c) / (a + b + c) + (d / 2) * (
            2 * a + b + c
        ) / (b + c + d)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = p
        return self._normalize(similarity_)

    @register("hawkins_dotson", to_distance=lambda x: 1 - x)
    def _get_hawkins_dotson(self, mol1_descriptor, mol2_descriptor):
        """Calculate Hawkins-Dotson similarity between two molecules.
        This is defined for two binary arrays as:
        Hawkins-Dotson similarity = 0.5 * (a / (a + b + c)
                                           + d / (d + b + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Hawkins-Dotson similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Hawkins-Dotson similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        similarity_ = 0.5 * (
            a / (a + b + c + SMALL_NUMBER) + d / (d + b + c + SMALL_NUMBER)
        )
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("jaccard")
    def _get_jaccard(self, mol1_descriptor, mol2_descriptor):
        """Calculate jaccard similarity between two molecules.
        This is defined for two binary arrays as:
        jaccard similarity = 3*a / (3*a + b + c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Jaccard similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Jaccard similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = 3 * a / (3 * a + b + c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("kulczynski")
    def _get_kulczynski(self, mol1_descriptor, mol2_descriptor):
        """Calculate kulczynski similarity between two molecules.
        This is defined for two binary arrays as:
        kulczynski similarity = 0.5 * a / ((a + b) + (a + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Kulczynski similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Kulczynski similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = 0.5 * a / ((a + b) + (a + c))
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("michael")
    def _get_michael(self, mol1_descriptor, mol2_descriptor):
        """Calculate michael similarity between two molecules.
        This is defined for two binary arrays as:
        michael similarity = 4*(a*d - b*c) / ((a + d)**2 + (b + c)**2)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Michael similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Michael similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p or (b + c) < SMALL_NUMBER:
            return 1.0
        similarity_ = 4 * (a * d - b * c) / ((a + d) ** 2 + (b + c) ** 2)
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("maxwell_pilliner", to_distance=lambda x: 1 - x)
    def _get_maxwell_pilliner(self, mol1_descriptor, mol2_descriptor):
        """Calculate Maxwell-Pilliner similarity between two molecules.
        This is defined for two binary arrays as:
        Maxwell-Pilliner similarity =
            2*(a*d - b*c) / ((a + b)*(c + d) + (a + c)*(b + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Maxwell-Pilliner similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Maxwell-Pilliner similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        denominator = (a + b) * (c + d) + (a + c) * (b + d)
        if a == p or d == p:
            return 1.0
        if denominator < SMALL_NUMBER:
            return 0.0

        similarity_ = 2 * (a * d - b * c) / denominator
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("mountford", to_distance=lambda x: 1 - x)
    def _get_mountford(self, mol1_descriptor, mol2_descriptor):
        """Calculate mountford similarity between two molecules.
        This is defined for two binary arrays as:
        mountford similarity = 2*a / (a*b + a*c + 2*b*c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Mountford similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Mountford similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        denominator = a * b + a * c + 2 * b * c
        if denominator < SMALL_NUMBER:
            return a / p
        similarity_ = 2 * a / denominator
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("pearson_heron", "pearson-heron")
    def _get_pearson_heron(self, mol1_descriptor, mol2_descriptor):
        """Calculate Pearson-Heron similarity between two molecules.
        This is defined for two binary arrays as:
        Pearson-Heron similarity =
         (a*d - b*c)/sqrt((a + b)*(a + c)*(b + d)*(c + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Pearson-Heron similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Pearson-Heron similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        if b == p or c == p:
            return 0.0
        denominator_ = np.sqrt(
            np.int64(a + b) * np.int64(a + c) * np.int64(b + d) * np.int64(c + d)
        )
        if denominator_ < SMALL_NUMBER:
            return 0.0
        similarity_ = (a * d - b * c) / denominator_
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("peirce_1", "peirce-1")
    def _get_peirce_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Peirce(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Peirce(1) similarity = (a*d - b*c) / ((a + b)*(c + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Peirce(1) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Peirce(1) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        if b == p or c == p:
            return 0.0
        similarity_ = (a * d - b * c) / ((a + b) * (c + d) + SMALL_NUMBER)
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("peirce_2", "peirce-2")
    def _get_peirce_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Peirce(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Peirce(2) similarity = (a*d - b*c) / ((a + c)*(b + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Peirce(2) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Peirce(2) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        if b == p or c == p:
            return 0.0
        similarity_ = (a * d - b * c) / ((a + c) * (b + d) + SMALL_NUMBER)
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("rogers-tanimoto", to_distance=lambda x: 1 - x)
    def _get_rogers_tanimoto(self, mol1_descriptor, mol2_descriptor):
        """Calculate rogers-tanimoto similarity between two molecules.
        This is defined for two binary arrays as:
        Rogers-Tanimoto similarity = (a + d) / (p + b + c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Rogers-Tanimoto similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Rogers-Tanimoto similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (a + d) / (p + b + c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("rogot_goldberg", to_distance=lambda x: 1 - x)
    def _get_rogot_goldberg(self, mol1_descriptor, mol2_descriptor):
        """Calculate Rogot-Goldberg similarity between two molecules.
        This is defined for two binary arrays as:
        Rogot-Goldberg similarity = (a / (2*a + b + c)) + (d / (2*d + b + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Rogot-Goldberg  similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Rogot-Goldberg similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        similarity_ = (a / (2 * a + b + c)) + (d / (2 * d + b + c))
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("russel-rao", to_distance=lambda x: 1 - x)
    def _get_russel_rao(self, mol1_descriptor, mol2_descriptor):
        """Calculate russel-rao similarity between two molecules.
        This is defined for two binary arrays as:
        Russel-Rao = a / p

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Russel-Rao similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Russel-Rao similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = a / p
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("simple_matching", "sokal-michener", "rand", to_distance=lambda x: 1 - x)
    def _get_simple_matching(self, mol1_descriptor, mol2_descriptor):
        """Calculate simple matching similarity between two molecules.
        This is defined for two binary arrays as:
        Simple Matching = (a + d) / p

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Simple Matching similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Simple Matching similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (a + d) / p
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("simpson")
    def _get_simpson(self, mol1_descriptor, mol2_descriptor):
        """Calculate simpson similarity between two molecules.
        This is defined for two binary arrays as:
        Simpson Similarity = a / min{(a + b), (a + c)}

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Simpson similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Simpson similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if min((a + b), (a + c)) < SMALL_NUMBER or a < SMALL_NUMBER:
            return 0.0
        similarity_ = a / min((a + b), (a + c))
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("sokal-sneath", "sokal-sneath_1", to_distance=lambda x: 1 - x)
    def _get_sokal_sneath(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sokal-Sneath similarity between two molecules.
        This is defined for two binary arrays as:
        Sokal-Sneath similarity = a / (a + 2*b + 2*c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Sokal-Sneath similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sokal-Sneath similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, _ = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = a / (a + 2 * b + 2 * c)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register(
        "symmetric_sokal_sneath",
        "sokal-sneath_2",
        "sokal-sneath-2",
        "symmetric-sokal-sneath",
    )
    def _get_symmetric_sokal_sneath(self, mol1_descriptor, mol2_descriptor):
        """Calculate Symmetric Sokal-Sneath similarity between two molecules.
        This is defined for two binary arrays as:
        Symmetric Sokal-Sneath similarity = (2*a + 2*d) / (p + a + d)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Symmetric Sokal-Sneath similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Symmetric Sokal-Sneath similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        similarity_ = (2 * a + 2 * d) / (p + a + d)
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("sokal-sneath-3", "sokal-sneath_3", to_distance=lambda x: 1 - x)
    def _get_sokal_sneath_3(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sokal-Sneath(3) similarity between two molecules.
        This is defined for two binary arrays as:
        Sokal-Sneath(3) similarity =
        (1/4) * (a/(a + b) + a/(a + c) + d/(b + d) + d/(c + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Sokal-Sneath(3) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sokal-Sneath(3) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        if a < SMALL_NUMBER and d < SMALL_NUMBER:
            return 0.0
        similarity_ = (1 / 4) * (
            a / (a + b + SMALL_NUMBER)
            + a / (a + c + SMALL_NUMBER)
            + d / (b + d + SMALL_NUMBER)
            + d / (c + d + SMALL_NUMBER)
        )
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("sokal-sneath-4", "sokal-sneath_4", to_distance=lambda x: 1 - x)
    def _get_sokal_sneath_4(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sokal-Sneath(4) similarity between two molecules.
        This is defined for two binary arrays as:
        Sokal-Sneath(4) similarity =
         a/sqrt((a + b) * (a + c)) * d/sqrt((b + d) *(c + d))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Sokal-Sneath(4) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sokal-Sneath(4) similarity is only useful for bit strings "
                "generated from fingerprints. Consider using "
                "other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        if a == p or d == p:
            return 1.0
        if a < SMALL_NUMBER or d < SMALL_NUMBER:
            return 0.0
        similarity_ = (
            a
            / (np.sqrt((a + b) * (a + c)) + SMALL_NUMBER)
            * d
            / (np.sqrt((b + d) * (c + d) + SMALL_NUMBER))
        )

        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("sorgenfrei")
    def _get_sorgenfrei(self, mol1_descriptor, mol2_descriptor):
        """Calculate Sorgenfrei similarity between two molecules.
        This is defined for two binary arrays as:
        Sorgenfrei similarity = a**2 / ((a + b)*(a + c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Sorgenfrei similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Sorgenfrei similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        if a < SMALL_NUMBER:
            return 0.0
        similarity_ = a**2 / ((a + b) * (a + c))
        self.normalize_fn["shift_"] = 0.0
        self.normalize_fn["scale_"] = 1.0
        return self._normalize(similarity_)

    @register("yule_1", "yule-1")
    def _get_yule_1(self, mol1_descriptor, mol2_descriptor):
        """Calculate Yule(1) similarity between two molecules.
        This is defined for two binary arrays as:
        Symmetric Yule(1) similarity = (a*d - b*c) / (a*d + b*c)

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Yule(1) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Yule(1) similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        denominator = a * d + b * c + SMALL_NUMBER
        if a == p or d == p or b * c < SMALL_NUMBER:
            return 1.0
        similarity_ = (a * d - b * c) / denominator
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    @register("yule_2", "yule-2", to_distance=lambda x: 1 - x)
    def _get_yule_2(self, mol1_descriptor, mol2_descriptor):
        """Calculate Yule(2) similarity between two molecules.
        This is defined for two binary arrays as:
        Symmetric Yule(2) similarity = (sqrt(a*d) - sqrt(b*c))
                                       / (sqrt(a*d) + sqrt(b*c))

        Args:
            mol1_descriptor (AIMSim.ops Descriptor)
            mol2_descriptor (AIMSim.ops Descriptor)

        Returns:
            (float): Yule(2) similarity value
        """
        if not (mol1_descriptor.is_fingerprint() and mol2_descriptor.is_fingerprint()):
            raise ValueError(
                "Yule(2) similarity is only useful "
                "for bit strings generated from fingerprints. Consider "
                "using other similarity measures for arbitrary vectors."
            )
        a, b, c, d = self._get_abcd(mol1_descriptor, mol2_descriptor)
        p = a + b + c + d
        denominator = np.sqrt(a * d) + np.sqrt(b * c) + SMALL_NUMBER
        if a == p or d == p or b * c < SMALL_NUMBER:
            return 1.0
        similarity_ = (np.sqrt(a * d) - np.sqrt(b * c)) / denominator
        self.normalize_fn["shift_"] = 1.0
        self.normalize_fn["scale_"] = 2.0
        return self._normalize(similarity_)

    def _get_abcd(self, fingerprint1, fingerprint2):
        """Get a, b, c, d, where:
        a = #bits(bits(array 1) and bits(array 2))
        b = #bits(bits(array 1) and bits(~array 2))
        c = #bits(bits(~array 1) and bits(array 2))
        d = #bits(bits(~array 1) and bits(~array 2)) // "~": complement operator
        p = a + b + c + d = bits(array 1 or array 2)

        Args:
            fingerprint1 (Descriptor)
            fingerprint2 (Descriptor)

        Returns:
            (tuple): (a, b, c, d)

        Note:
            If arrays of unequal lengths are passed, the larger array is folded
            to the length of the smaller array.

        """
        arr1, arr2 = Descriptor.fold_to_equal_length(fingerprint1, fingerprint2)
        not_arr1 = np.logical_not(arr1)
        not_arr2 = np.logical_not(arr2)
        a = np.sum(arr1 & arr2)
        b = np.sum(arr1 & not_arr2)
        c = np.sum(not_arr1 & arr2)
        d = np.sum(not_arr1 & not_arr2)
        assert (a + b + c + d) == arr1.size == arr2.size
        return a, b, c, d

    def _normalize(self, similarity_):
        return (similarity_ + self.normalize_fn["shift_"]) / self.normalize_fn["scale_"]

    def is_distance_metric(self):
        """Check if the similarity measure is a distance metric.

        Returns:
            bool: True if it is a distance metric.
        """
        return self._is_distance

    @staticmethod
    @lru_cache(maxsize=None)
    def _validate_fprint(descriptor: Descriptor) -> bool:
        """Are there any non-zero bits in this descriptor?

        Args:
            descriptor (Descriptor): AIMSim.ops.Descriptor object

        Returns:
            boolean: False if descriptor is all zero, True otherwise
        """
        return np.any(descriptor.to_numpy())

    @staticmethod
    def get_compatible_metrics():
        """Return a dictionary with which types of metrics each fingerprint supports.

        Returns:
            dict: comptabile FP's: metrics
        """
        out = {}
        fprints = Descriptor.get_all_supported_descriptors()
        for fp in fprints:
            if fp in [
                "morgan_fingerprint",
                "topological_fingerprint",
                "daylight_fingerprint",
                "maccs_keys",
            ]:  # explicit bit vectors
                out[fp] = SimilarityMeasure.get_supported_binary_metrics()
            elif fp in ["atom-pair_fingerprint", "torsion_fingerprint"]:
                # int vectors
                out[fp] = SimilarityMeasure.get_supported_metrics()
            else:  # mordred descriptors, custom descriptors
                out[fp] = SimilarityMeasure.get_supported_general_metrics()
        return out

    @staticmethod
    def get_supported_general_metrics():
        """Return a list of strings for the currently implemented
        similarity measures, aka metrics, which support vectors other
        then binary vectors.

        Returns:
            List: List of strings.
        """
        return list(
            set(SimilarityMeasure.get_supported_metrics())
            - set(SimilarityMeasure.get_supported_binary_metrics())
        )

    @staticmethod
    def get_supported_binary_metrics():
        """Return a list of strings for the currently implemented
        similarity measures, aka metrics, which only support binary
        vectors.

        Returns:
            List: List of strings.
        """
        return BINARY_METRICS

    @staticmethod
    def get_supported_metrics():
        """Return a list of strings for the currently implemented
        similarity measures, aka metrics.

        Returns:
            List: List of strings.
        """
        return ALL_METRICS

    @staticmethod
    def get_uniq_metrics():
        """Return a list of strings for the currently implemented
        similarity measures, aka metrics. Each unique similarity metric is
        uniquely represented with redundant tags removed.

        Returns:
            List: List of strings.
        """
        return UNIQUE_METRICS

    def __str__(self):
        return self.label_
