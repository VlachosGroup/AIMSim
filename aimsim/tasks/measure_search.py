"""Identify most appropriate choice of fingerprint and similarity measure by
evaluating the response of the nearest and furthest neighbors. This is called
measure choice for brevity (although both measure and features are chosen)"""
from collections import namedtuple
import json
from os import makedirs
from os.path import dirname

import numpy as np

from aimsim.chemical_datastructures import MoleculeSet
from aimsim.exceptions import InvalidConfigurationError
from aimsim.ops import Descriptor, SimilarityMeasure
from aimsim.utils.plotting_scripts import plot_multiple_barchart
from .task import Task
from .see_property_variation_with_similarity \
    import SeePropertyVariationWithSimilarity


class MeasureSearch(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()
        configs.update(kwargs)
        super().__init__(configs)
        self.plot_settings = None
        self.log_fpath = None
        self._extract_configs()

    def _extract_configs(self):
        """
        Raises:
        InvalidConfigurationError: If correlation_type does not match
                                   implemented types.
        """
        self.plot_settings = {'colors': ['#78DEC7', '#FB7AFC', '#FBC7F7'],
                              'xticksize': 10,
                              'yticksize': 20,
                              'title': "",
                              'title_fontsize': 24,
                              'xlabel_fontsize': 20,
                              'ylabel_fontsize': 20,
                              }
        self.plot_settings.update(self.configs.get("plot_settings",
                                                   {}))
        try:
            self.prop_var_w_similarity = SeePropertyVariationWithSimilarity(
                correlation_type=self.configs.get('correlation_type'))
        except InvalidConfigurationError as e:
            raise e
        self.log_fpath = self.configs.get("log_file_path", None)
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

    def __call__(self,
                 molecule_set_configs,
                 fingerprint_type=None,
                 fingerprint_params=None,
                 similarity_measure=None,
                 subsample_subset_size=0.01,
                 optim_algo='max_min',
                 show_top=0,
                 only_metric=True,
                 ):
        """
        Calculate the correlation in the properties of molecules in set
        and their nearest and furthest neighbors using different
        fingerprints / similarity measure choices. Choose the best fingerprint
        and similarity measure pair (called measure choice for brevity)
        based on an optimization strategy.

        Args:
            molecule_set_configs (dict): All configurations (except
                fingerprint_type, fingerprint_params and similarity_measure)
                needed to form the moleculeSet.
            fingerprint_type (str): Label to indicate which fingerprint to
                use. If supplied, fingerprint is fixed and optimization
                carried out over similarity measures. Use None to indicate
                that optimization needs to be carried out over fingerprints.
                Default is None.
            fingerprint_params (dict): Hyper-parameters for fingerprints.
                Passed to the MoleculeSet constructor. If None is passed,
                set to empty dictionary before passing to MoleculeSet.
            similarity_measure (str): Label to indicate which similarity
                measure to use. If supplied, similarity measure is fixed
                and optimization carried out over similarity measures.
                Use None to indicate that optimization needs to be carried
                out over fingerprints. Default is None.
            subsample_subset_size (float): Fraction of molecule_set to
                subsample. This is separate from the sample_ratio parameter
                used when creating a moleculeSet since it is recommended
                to have an more aggressive subsampling strategy for this task
                due to the combinatorial explosion of looking at multiple
                fingerprints and similarity measures. Default is 0.01.
            optim_algo (str): Label to indicate the optimization algorithm
                chosen. Options are:
                'max': The measure choice which maximizes correlation
                    of properties between nearest neighbors (most similar).
                    This is the default.
                'min': The measure choice which minimizes the absolute value
                    of property correlation
                    between furthest neighbors (most dissimilar).
                'max_min': The measure choice which maximizes correlation
                    of properties between nearest neighbors (most similar)
                    and minimizes he absolute value of property correlation
                    between furthest neighbors (most dissimilar).
                    This is the default.
            show_top (int): Number of top performing measures to show in plot.
                If 0, no plots are generated and the top performer is returned.
            only_metric (bool): If True only similarity measures satisfying
                the metricity property
                (i.e. can be converted to distance metrics) are selected.

        Returns:
            (NamedTuple): Top performer with fields:
                fingerprint_type (str): Label for fingerprint type
               similarity_measure (str): Label for similarity measure
               nearest_neighbor_correlation (float): Correlation of property
                   of molecule and its nearest neighbor.
               furthest_neighbor_correlation (float): Correlation of property
                   of molecule and its furthest neighbor.
               score_ (float): Overall score based on optimization strategy.
                   More is better.

        """
        print(f'Using subsample size {subsample_subset_size} for '
              f'measure search')
        trial_ = namedtuple('trial_', ['fingerprint_type',
                                       'similarity_measure',
                                       'nearest_neighbor_correlation',
                                       'furthest_neighbor_correlation',
                                       'score_'])
        if fingerprint_type is None:
            all_fingerprint_types = Descriptor.get_supported_fprints()
            fingerprint_params = None
        else:
            all_fingerprint_types = [fingerprint_type]
        if similarity_measure is None:
            if only_metric:
                print('Only trying measures with valid distance metrics')
            all_similarity_measures = SimilarityMeasure.get_uniq_metrics()
        else:
            all_similarity_measures = [similarity_measure]
        is_verbose = molecule_set_configs.get("is_verbose", False)
        all_scores = []
        if fingerprint_params is None:
            fingerprint_params = {}
        for similarity_measure in all_similarity_measures:
            if only_metric and not SimilarityMeasure(
                    metric=similarity_measure).is_distance_metric():
                continue
            if is_verbose:
                print(f'Trying {similarity_measure} similarity')
            for fingerprint_type in all_fingerprint_types:
                if is_verbose:
                    print(f'Trying {fingerprint_type} fingerprint')
                try:
                    molecule_set = MoleculeSet(
                        molecule_database_src=molecule_set_configs[
                            'molecule_database_src'],
                        molecule_database_src_type=molecule_set_configs[
                            'molecule_database_src_type'],
                        similarity_measure=similarity_measure,
                        fingerprint_type=fingerprint_type,
                        fingerprint_params=fingerprint_params,
                        is_verbose=is_verbose,
                        n_threads=molecule_set_configs.get(
                            'n_threads', 1),
                        sampling_ratio=subsample_subset_size)
                except (InvalidConfigurationError, ValueError, RuntimeError) as e:
                    if is_verbose:
                        print(f'Could not try {fingerprint_type} with '
                              f'similarity measure {similarity_measure} due to '
                              f'{e}')
                    continue
                nearest_corr, nearest_p_val = self.prop_var_w_similarity. \
                    get_property_correlations_in_most_similar(
                        molecule_set)
                furthest_corr, furthest_p_val = self.prop_var_w_similarity. \
                    get_property_correlations_in_most_dissimilar(
                        molecule_set)
                if optim_algo == 'max_min':
                    score_ = nearest_corr - abs(furthest_corr)
                elif optim_algo == 'max':
                    score_ = nearest_corr
                elif optim_algo == 'min':
                    score_ = -abs(furthest_corr)
                else:
                    raise InvalidConfigurationError(f'{optim_algo} '
                                                    f'not implemented')
                all_scores.append(trial_(
                    fingerprint_type=fingerprint_type,
                    similarity_measure=similarity_measure,
                    nearest_neighbor_correlation=nearest_corr,
                    furthest_neighbor_correlation=furthest_corr,
                    score_=score_))
        all_scores.sort(key=lambda x: x[-1], reverse=True)
        if self.log_fpath is not None:
            print('Writing to ', self.log_fpath)
            log_data = [trial._asdict() for trial in all_scores]
            with open(self.log_fpath, "w") as fp:
                json.dump(log_data, fp)

        if show_top > 0:
            top_performers = all_scores[:show_top]
            all_nearest_neighbor_correlations = []
            all_furthest_neighbor_correlations = []
            top_scores = []
            all_measures = []
            for trial in top_performers:
                all_nearest_neighbor_correlations.append(
                    trial.nearest_neighbor_correlation)
                all_furthest_neighbor_correlations.append(
                    trial.furthest_neighbor_correlation)
                top_scores.append(trial.score_)
                all_measures.append(Descriptor.shorten_label(
                    trial.fingerprint_type)
                    + '\n'
                    + trial.similarity_measure)
            bar_heights = np.array([top_scores,
                                    all_nearest_neighbor_correlations,
                                    all_furthest_neighbor_correlations])
            colors = self.plot_settings.pop('colors')
            plot_multiple_barchart(x=[_ for _ in range(len(top_performers))],
                                   heights=bar_heights,
                                   legend_labels=['Overall scores',
                                                  'Nearest neighbor property '
                                                  'correlation',
                                                  'Furthest neighbor property '
                                                  'correlations'],
                                   colors=colors,
                                   xtick_labels=all_measures,
                                   ylabel='Value',
                                   xlabel='Measure',
                                   **self.plot_settings)

        return all_scores[0]

    def __str__(self):
        return "Task: determine appropriate fingerprint type and " \
               "similarity measure for property of interest"

    def get_best_measure(self,
                         molecule_set_configs,
                         fingerprint_type=None,
                         similarity_measure=None,
                         subsample_subset_size=0.01,
                         optim_algo='max_min',
                         only_metric=False,
                         show_top=0,
                         ):
        """Get the best measure for quantity of interest.

        Args:
            molecule_set_configs (dict): All configurations (except
                fingerprint_type and similarity_measure) needed to form
                the moleculeSet.
            fingerprint_type (str): Label to indicate which fingerprint to
                use. If supplied, fingerprint is fixed and optimization
                carried out over similarity measures. Use None to indicate
                that optimization needs to be carried out over fingerprints.
                Default is None.
            similarity_measure (str): Label to indicate which similarity
                measure to use. If supplied, similarity measure is fixed
                and optimization carried out over similarity measures.
                Use None to indicate that optimization needs to be carried
                out over fingerprints. Default is None.
            subsample_subset_size (float): Fraction of molecule_set to
                subsample. This is separate from the sample_ratio parameter
                used when creating a moleculeSet since it is recommended
                to have an more aggressive subsampling strategy for this task
                due to the combinatorial explosion of looking at multiple
                fingerprints and similarity measures. Default is 0.01.
            optim_algo (str): Label to indicate the optimization algorithm
                chosen. Options are:
                'max': The measure choice which maximizes correlation
                    of properties between nearest neighbors (most similar).
                    This is the default.
                'min': The measure choice which minimizes the absolute value of
                    property correlation between furthest neighbors
                    (most dissimilar).
                'max_min': The measure choice which maximizes correlation
                    of properties between nearest neighbors (most similar)
                    and minimizes he absolute value of property correlation
                    between furthest neighbors (most dissimilar).
                    This is the default.
            only_metric (bool): If True only similarity measures satisfying
                the metricity property
                (i.e. can be converted to distance metrics) are selected.

        Returns:
            (NamedTuple): Top performer with fields:
                fingerprint_type (str): Label for fingerprint type
               similarity_measure (str): Label for similarity measure
               nearest_neighbor_correlation (float): Correlation of property
                   of molecule and its nearest neighbor.
               furthest_neighbor_correlation (float): Correlation of property
                   of molecule and its furthest neighbor.
               score_ (float): Overall score based on optimization strategy.
                   More is better.

        """
        return self.__call__(
            molecule_set_configs,
            fingerprint_type=fingerprint_type,
            similarity_measure=similarity_measure,
            subsample_subset_size=subsample_subset_size,
            optim_algo=optim_algo,
            only_metric=only_metric,
            show_top=show_top,
        )
