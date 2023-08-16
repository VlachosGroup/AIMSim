"""
Calculates the Extended Similarity Indexes as shown in
this table: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3/tables/1
and described in "Extended similarity indices: the benefits of comparing more than two objects
simultaneously. Part 1: Theory and characteristics"

Both gen_sim_dict and calculate_counters were provided by Raymond Quintana, similar to
that which is here: https://github.com/ramirandaq/MultipleComparisons
"""
import json
from math import log, ceil

import numpy as np

from .task import Task


class ExtendedSimilarityIndices(Task):
    def __call__(self, molecule_set):
        """
        Calculates the extended similarity indices for the given molecule_set.

        Args:
            molecule_set (AIMSim.chemical_datastructures Molecule): Target
                molecule.

        """
        n_mols = len(molecule_set.molecule_database)
        fprints = []
        for i in range(n_mols):
            fprints.append(molecule_set.molecule_database[i].get_descriptor_val())
        fprint_array_sum = np.sum(fprints, axis=0)
        indices_array = self.gen_sim_dict(fprint_array_sum, n_mols)

        print("Extended Similarity Indices:")
        print(json.dumps(indices_array, indent=2))
        return indices_array

    def _extract_configs(self):
        pass

    def __str__(self):
        return "Task: Calculate Extended Similarity Indices"

    def gen_sim_dict(
        self, c_total, n_fingerprints, c_threshold=None, w_factor="fraction"
    ):
        # Indices
        # AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
        # Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
        # JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
        # SM: Sokal-Michener, SSn: Sokal-Sneath n

        counters = self.calculate_counters(
            c_total, n_fingerprints, c_threshold=c_threshold, w_factor="fraction"
        )

        # Weighted Indices
        ac_w = (2 / np.pi) * np.arcsin(
            np.sqrt(counters["total_w_sim"] / counters["w_p"])
        )
        bub_w = ((counters["w_a"] * counters["w_d"]) ** 0.5 + counters["w_a"]) / (
            (counters["w_a"] * counters["w_d"]) ** 0.5
            + counters["w_a"]
            + counters["total_w_dis"]
        )
        ct1_w = (log(1 + counters["w_a"] + counters["w_d"])) / (
            log(1 + counters["w_p"])
        )
        ct2_w = (log(1 + counters["w_p"]) - log(1 + counters["total_w_dis"])) / (
            log(1 + counters["w_p"])
        )
        ct3_w = (log(1 + counters["w_a"])) / (log(1 + counters["w_p"]))
        ct4_w = (log(1 + counters["w_a"])) / (
            log(1 + counters["w_a"] + counters["total_w_dis"])
        )
        fai_w = (counters["w_a"] + 0.5 * counters["w_d"]) / (counters["w_p"])
        gle_w = (2 * counters["w_a"]) / (2 * counters["w_a"] + counters["total_w_dis"])
        ja_w = (3 * counters["w_a"]) / (3 * counters["w_a"] + counters["total_w_dis"])
        ja0_w = (3 * counters["total_w_sim"]) / (
            3 * counters["total_w_sim"] + counters["total_w_dis"]
        )
        jt_w = (counters["w_a"]) / (counters["w_a"] + counters["total_w_dis"])
        rt_w = (counters["total_w_sim"]) / (counters["w_p"] + counters["total_w_dis"])
        rr_w = (counters["w_a"]) / (counters["w_p"])
        sm_w = (counters["total_w_sim"]) / (counters["w_p"])
        ss1_w = (counters["w_a"]) / (counters["w_a"] + 2 * counters["total_w_dis"])
        ss2_w = (2 * counters["total_w_sim"]) / (
            counters["w_p"] + counters["total_w_sim"]
        )

        # Non-Weighted Indices
        ac_nw = (2 / np.pi) * np.arcsin(
            np.sqrt(counters["total_w_sim"] / counters["p"])
        )
        bub_nw = ((counters["w_a"] * counters["w_d"]) ** 0.5 + counters["w_a"]) / (
            (counters["a"] * counters["d"]) ** 0.5
            + counters["a"]
            + counters["total_dis"]
        )
        ct1_nw = (log(1 + counters["w_a"] + counters["w_d"])) / (log(1 + counters["p"]))
        ct2_nw = (log(1 + counters["w_p"]) - log(1 + counters["total_w_dis"])) / (
            log(1 + counters["p"])
        )
        ct3_nw = (log(1 + counters["w_a"])) / (log(1 + counters["p"]))
        ct4_nw = (log(1 + counters["w_a"])) / (
            log(1 + counters["a"] + counters["total_dis"])
        )
        fai_nw = (counters["w_a"] + 0.5 * counters["w_d"]) / (counters["p"])
        gle_nw = (2 * counters["w_a"]) / (2 * counters["a"] + counters["total_dis"])
        ja_nw = (3 * counters["w_a"]) / (3 * counters["a"] + counters["total_dis"])
        ja0_nw = (3 * counters["total_w_sim"]) / (
            3 * counters["total_sim"] + counters["total_dis"]
        )
        jt_nw = (counters["w_a"]) / (counters["a"] + counters["total_dis"])
        rt_nw = (counters["total_w_sim"]) / (counters["p"] + counters["total_dis"])
        rr_nw = (counters["w_a"]) / (counters["p"])
        sm_nw = (counters["total_w_sim"]) / (counters["p"])
        ss1_nw = (counters["w_a"]) / (counters["a"] + 2 * counters["total_dis"])
        ss2_nw = (2 * counters["total_w_sim"]) / (counters["p"] + counters["total_sim"])

        # Dictionary with all the results
        Indices = {
            "nw": {
                "AC": ac_nw,
                "BUB": bub_nw,
                "CT1": ct1_nw,
                "CT2": ct2_nw,
                "CT3": ct3_nw,
                "CT4": ct4_nw,
                "Fai": fai_nw,
                "Gle": gle_nw,
                "Ja": ja_nw,
                "Ja0": ja0_nw,
                "JT": jt_nw,
                "RT": rt_nw,
                "RR": rr_nw,
                "SM": sm_nw,
                "SS1": ss1_nw,
                "SS2": ss2_nw,
            },
            "w": {
                "AC": ac_w,
                "BUB": bub_w,
                "CT1": ct1_w,
                "CT2": ct2_w,
                "CT3": ct3_w,
                "CT4": ct4_w,
                "Fai": fai_w,
                "Gle": gle_w,
                "Ja": ja_w,
                "Ja0": ja0_w,
                "JT": jt_w,
                "RT": rt_w,
                "RR": rr_w,
                "SM": sm_w,
                "SS1": ss1_w,
                "SS2": ss2_w,
            },
        }
        return Indices

    def calculate_counters(
        self, c_total, n_fingerprints, c_threshold=None, w_factor="fraction"
    ):
        """Calculate 1-similarity, 0-similarity, and dissimilarity counters

        Arguments
        ---------
        c_total : np.ndarray
            Vector containing the sums of each column of the fingerprint matrix.

        n_fingerprints : int
            Number of objects to be compared.

        c_threshold : {None, 'dissimilar', int}
            Coincidence threshold.
            None : Default, c_threshold = n_fingerprints % 2
            'dissimilar' : c_threshold = ceil(n_fingerprints / 2)
            int : Integer number < n_fingerprints
            float : Real number in the (0 , 1) interval. Indicates the % of the total data that will serve as threshold.

        w_factor : {"fraction", "power_n"}
            Type of weight function that will be used.
            'fraction' : similarity = d[k]/n
                        dissimilarity = 1 - (d[k] - n_fingerprints % 2)/n_fingerprints
            'power_n' : similarity = n**-(n_fingerprints - d[k])
                        dissimilarity = n**-(d[k] - n_fingerprints % 2)
            other values : similarity = dissimilarity = 1

        Returns
        -------
        counters : dict
            Dictionary with the weighted and non-weighted counters.

        Notes
        -----
        Please, cite the original papers on the n-ary indices:
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4
        """
        # Assign c_threshold
        if not c_threshold:
            c_threshold = n_fingerprints % 2
        if c_threshold == "dissimilar":
            c_threshold = ceil(n_fingerprints / 2)
        if c_threshold == "min":
            c_threshold = n_fingerprints % 2
        if isinstance(c_threshold, int):
            if c_threshold >= n_fingerprints:
                raise ValueError(
                    "c_threshold cannot be equal or greater than n_fingerprints."
                )
            c_threshold = c_threshold
        if 0 < c_threshold < 1:
            c_threshold *= n_fingerprints

        # Set w_factor
        if w_factor:
            if "power" in w_factor:
                power = int(w_factor.split("_")[-1])

                def f_s(d):
                    return power ** -float(n_fingerprints - d)

                def f_d(d):
                    return power ** -float(d - n_fingerprints % 2)

            elif w_factor == "fraction":

                def f_s(d):
                    return d / n_fingerprints

                def f_d(d):
                    return 1 - (d - n_fingerprints % 2) / n_fingerprints

            else:

                def f_s(d):
                    return 1

                def f_d(d):
                    return 1

        else:

            def f_s(d):
                return 1

            def f_d(d):
                return 1

        # Calculate a, d, b + c

        a_indices = 2 * c_total - n_fingerprints > c_threshold
        d_indices = n_fingerprints - 2 * c_total > c_threshold
        dis_indices = np.abs(2 * c_total - n_fingerprints) <= c_threshold

        a = np.sum(a_indices)
        d = np.sum(d_indices)
        total_dis = np.sum(dis_indices)

        a_w_array = f_s(2 * c_total[a_indices] - n_fingerprints)
        d_w_array = f_s(abs(2 * c_total[d_indices] - n_fingerprints))
        total_w_dis_array = f_d(abs(2 * c_total[dis_indices] - n_fingerprints))

        w_a = np.sum(a_w_array)
        w_d = np.sum(d_w_array)
        total_w_dis = np.sum(total_w_dis_array)

        total_sim = a + d
        total_w_sim = w_a + w_d
        p = total_sim + total_dis
        w_p = total_w_sim + total_w_dis

        counters = {
            "a": a,
            "w_a": w_a,
            "d": d,
            "w_d": w_d,
            "total_sim": total_sim,
            "total_w_sim": total_w_sim,
            "total_dis": total_dis,
            "total_w_dis": total_w_dis,
            "p": p,
            "w_p": w_p,
        }
        return counters
