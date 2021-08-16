"""Identify most appropriate choice of fingerprint and similarity measure by
evaluating the response of the nearest and furthest neighbors. This is called
measure choice for brevity (although both measure and features are chosen)"""
from collections import namedtuple

import pylustrator

from molSim.chemical_datastructures import MoleculeSet
from molSim.exceptions import InvalidConfigurationError
from molSim.ops import Descriptor, SimilarityMeasure
from molSim.utils.plotting_scripts import plot_multiple_barchart
from .task import Task
from .see_property_variation_with_similarity \
    import SeePropertyVariationWithSimilarity


pylustrator.start()


class MeasureSearch(Task):
    def __init__(self, configs=None, **kwargs):
        if configs is None:
            configs = dict()  # all configs are optional
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
        self.plot_settings = {'cmap': 'tab20'}
        self.plot_settings.update(self.configs.get("plot_settings",
                                                   {}))
        try:
            self.prop_var_w_similarity = SeePropertyVariationWithSimilarity(
                correlation_type=self.configs.get('correlation_type'))
        except InvalidConfigurationError as e:
            raise e
        if self.log_fpath is not None:
            log_dir = dirname(self.log_fpath)
            makedirs(log_dir, exist_ok=True)

    def __call__(self,
                 fingerprint_type=None,
                 similarity_measure=None,
                 subsample_subset_size=0.01,
                 optim_algo='max_min',
                 show_top=0,
                 only_metric=False,
                 **molecule_set_configs):
        """
        Calculate the correlation in the properties of molecules in set
        and their nearest and furthest neighbors using different
        fingerprints / similarity measure choices. Choose the best fingerprint
        and similarity measure pair (called measure choice for brevity)
        based on an optimization strategy.

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
                'min': The measure choice which minimizes property correlation
                    between furthest neighbors (most dissimilar).
                'max_min': The measure choice which maximizes correlation
                    of properties between nearest neighbors (most similar)
                    and minimizes property correlation between furthest
                    neighbors (most dissimilar). This is the default.
            show_top (int): Number of top performing measures to show in plot.
                If 0, no plots are generated and the top performer is returned.
            only_metrics (bool): If True only similarity measures satisfying
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
        trial_ = namedtuple('trial_', ['fingerprint_type',
                                       'similarity_measure',
                                       'nearest_neighbor_correlation',
                                       'furthest_neighbor_correlation',
                                       'score_'])
        if fingerprint_type is None:
            all_fingerprint_types = Descriptor.get_supported_fprints()
        else:
            all_fingerprint_types = [fingerprint_type]
        if similarity_measure is None:
            all_similarity_measures = SimilarityMeasure.get_supported_metrics()
        else:
            all_similarity_measures = [similarity_measure]
        is_verbose = molecule_set_configs.get("is_verbose", False)
        n_threads = molecule_set_configs.get("n_workers", 1)
        all_scores = []
        for similarity_measure in all_similarity_measures:
            if only_metric and SimilarityMeasure(
                    metric=similarity_measure).is_distance_metric():
                continue
            if is_verbose:
                print(f'Trying {similarity_measure} similarity')
            for fingerprint_type in all_fingerprint_types:
                if is_verbose:
                    print(f'Trying {fingerprint_type} fingerprint')
                molecule_set = MoleculeSet(
                    molecule_database_src=molecule_set_configs[
                        'molecule_database_src'],
                    molecule_database_src_type=molecule_set_configs[
                        'molecule_database_src_type'],
                    similarity_measure=similarity_measure,
                    fingerprint_type=fingerprint_type,
                    is_verbose=is_verbose,
                    n_threads=n_threads,
                    sampling_ratio=subsample_subset_size)

                nearest_corr, nearest_p_val = self.prop_var_w_similarity. \
                    get_property_correlations_in_most_similar(
                         molecule_set)
                furthest_corr, furthest_p_val = self.prop_var_w_similarity. \
                    get_property_correlations_in_most_dissimilar(
                        molecule_set)
                if optim_algo == 'max_min':
                    score_ = nearest_corr - furthest_corr
                elif optim_algo == 'max':
                    score_ = nearest_corr
                elif optim_algo == 'min':
                    score_ = -furthest_corr
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
        if show_top > 0:
            top_performers = all_scores[:show_top]
            all_nearest_neighbor_correlations = []
            all_furthest_neighbor_correlations = []
            all_scores = []
            all_measures = []
            for trial in top_performers:
                all_nearest_neighbor_correlations.append(
                    trial.nearest_neighbor_correlation)
                all_furthest_neighbor_correlations.append(
                    rial.furthest_neighbor_correlation)
                all_scores.append(trial.score_)
                all_measures.append(trial.fingerprint_type
                                    + ' '
                                    + trial.similarity_measure)
            bar_heights = [all_scores,
                           all_nearest_neighbor_correlations,
                           all_furthest_neighbor_correlations]
            plot_multiple_barchart(x=[_ for _ in range(top_performers)],
                                   heights=bar_heights,
                                   legend_labels=['Overall scores',
                                                  'Nearest neighbor property '
                                                  'correlation',
                                                  'Furthest neighbor property '
                                                  'correlations'],
                                   colors=[plt.cm.get_cmap(
                                               self.plot_settings['cmap'],
                                               len(bar_heights))(bar_id)
                                           for bar_id in range(bar_heights)])

        return all_scores[0]

    def __str__(self):
        return "Task: determine appropriate fingerprint type and " \
               "similarity measure for property of interest"
