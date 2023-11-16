import numpy as np
import re
import math
from itertools import product, permutations, combinations_with_replacement
from lingpy import Wordlist, Multiple, Alignments
from collections import defaultdict
from tqdm import tqdm

from shibboleth.utils.io import *


# TODO re-integrate function for caching confusion matrices
class ShibbolethCalculator(object):
    def __init__(self, varieties, data_fp, skip_concepts=None, skip_sites=None, realign=False):
        """
        initialize the calculator class; defining the cluster of varieties in question,
        loading in the lexical data, and providing a similarity matrix over all sounds for constructing alignments.
        :param varieties: the defined cluster of varieties to be investigated.
        :param cldf_meta_fp: the filepath to the CLDF metadata file corresponding to the dataset
        :param sim_mat_fp: the filepath to retrieve the similarity matrix for sound pairs from
        """
        # TODO make more flexible (aka pass down args and parse them)
        # parse arguments
        if skip_concepts is None:
            self.skip_concepts = []
        else:
            self.skip_concepts = skip_concepts

        if skip_sites is None:
            self.skip_sites = []
        else:
            self.skip_sites = skip_sites

        # load lexical data as wordlist
        self.data = load_data(data_fp)

        # store defined variety cluster by given matching strategy
        # TODO enable passing down different matching strategies
        self.varieties = []
        for var_in_data in self.data.cols:
            id = var_in_data.split("_")[0]
            if id in varieties:
                self.varieties.append(var_in_data)

        # assign an integer id to each sound and keep track of the global frequencies of each sound
        self.frequencies = defaultdict(int)
        next_token_id = 1
        self.enc_dict = {"-": 0}
        self.dec_dict = {0: "-"}

        for _, tokens in self.data.iter_rows("tokens"):
            for token in tokens:
                self.frequencies[token] += 1
                if token not in self.enc_dict:
                    self.enc_dict[token] = next_token_id
                    self.dec_dict[next_token_id] = token
                    next_token_id += 1

        alphabet_size = len(self.enc_dict)

        # initialize 3 confusion matrices:
        # one for inside the given cluster (int), one for all varieties outside the cluster (ext),
        # one for alignments between varieties from inside and outside the cluster (cross).
        # IMPORTANT: 'int' and 'ext' matrices must be symmetric, while 'cross' matrix is directed!
        self.int_conf_mat = np.zeros(shape=(alphabet_size, alphabet_size), dtype=int)
        self.ext_conf_mat = np.zeros(shape=(alphabet_size, alphabet_size), dtype=int)
        self.cross_conf_mat = np.zeros(shape=(alphabet_size, alphabet_size), dtype=int)

        # optionally construct alignments
        if realign:
            self.align()

        # populate those confusion matrices by counting correspondences
        self.count_phonetic_correspondences()

    def align(self):
        """
        construct multiple alignments for all cognate sets.
        """
        self.data.renumber("concept", "cogid", override=True)
        alms = Alignments(self.data, ref="cogid", transcription="tokens")
        alms.align()
        self.data = alms

    def count_phonetic_correspondences(self):
        # if no alignments are found in the data, generate alignments
        if "alignment" not in self.data.columns:
            self.align()

        # get indices corresponding to the site and the alignment columns
        site_idx = self.data.header.get("doculect", -1)
        alignment_idx = self.data.header.get("alignment", -1)

        if site_idx < 0 or alignment_idx < 0:
            raise ValueError

        # iterate over concepts (=cognate sets)
        for concept in tqdm(self.data.rows, desc="Counting phonetic correspondences..."):
            if concept in self.skip_concepts:
                continue
            # get all form IDs corresponding to the concept
            form_ids = self.data.get_list(row=concept, flat=True)
            # retrieve data for all possible form pairs within the concept
            for id1, id2 in combinations_with_replacement(form_ids, 2):
                form1 = self.data[id1]
                form2 = self.data[id2]

                # retrieve sites corresponding to the forms
                site1 = form1[site_idx]
                site2 = form2[site_idx]

                if site1 in self.skip_sites or site2 in self.skip_sites:
                    continue

                # retrieve the confusion matrix to write to
                matrix_is_symmetric = True
                if site1 in self.varieties and site2 in self.varieties:
                    relevant_matrix = self.int_conf_mat
                elif site1 not in self.varieties and site2 not in self.varieties:
                    relevant_matrix = self.ext_conf_mat
                else:
                    relevant_matrix = self.cross_conf_mat
                    matrix_is_symmetric = False
                    # make sure that 'form1' is inside the defined cluster and 'form2' is outside -
                    # flip it around if it is the other way
                    if site2 in self.varieties:
                        form1, form2 = form2, form1

                # iterate over the alignments and store counts to the respective matrix
                alm1 = form1[alignment_idx]
                alm2 = form2[alignment_idx]

                for token1, token2 in zip(alm1, alm2):
                    idx1 = self.enc_dict[token1]
                    idx2 = self.enc_dict[token2]
                    relevant_matrix[idx1][idx2] += 1
                    # make sure that the internal and external matrix stay symmetric by counting the correspondence
                    # in both directions
                    if matrix_is_symmetric:
                        relevant_matrix[idx2][idx1] += 1

    def calculate_dist(self, normalize=True):
        """
        calculate the distinctiveness for all sound pairs.
        :param normalize: calculate the Normalized PMI if True; (unnormalized) PMI otherwise.
        :return:
        """
        dist_per_combination = {}
        col_sums = self.cross_conf_mat.sum(axis=0)
        row_sums = self.cross_conf_mat.sum(axis=1)
        probs = self.cross_conf_mat / np.sum(self.cross_conf_mat)
        marginal_probs_rows = row_sums / np.sum(row_sums)
        marginal_probs_cols = col_sums / np.sum(col_sums)

        # NOTATION: internal [i] corresponds to external [j]
        for i, j in permutations(self.dec_dict.keys(), 2):
            char_pair = "[%s] : [%s]" % (self.dec_dict[i], self.dec_dict[j])
            prob_i = marginal_probs_rows[i]
            prob_j = marginal_probs_cols[j]
            prob_i_j = probs[i, j]

            # value can only be calculated for non-zero probabilities
            if (prob_i * prob_j * prob_i_j) != 0:
                dist = math.log(prob_i_j / (prob_i * prob_j))

                if normalize:
                    dist = dist / (- math.log(prob_i_j))
                dist_per_combination[char_pair] = dist
            else:
                dist_per_combination[char_pair] = 0

        # subtract PMI value for inverted relation; filter out correspondences with negative values
        dist_values = {}
        for pair in dist_per_combination:
            chars = re.findall("\[(.+?)]", pair)
            char1 = chars[0]
            char2 = chars[1]
            inv_pair = "[%s] : [%s]" % (char2, char1)
            if dist_per_combination[pair] > 0:
                if inv_pair in dist_per_combination:
                    dist_new = dist_per_combination[pair] - dist_per_combination[inv_pair]
                    dist_values[pair] = dist_new
                else:
                    dist_values[pair] = dist_per_combination[pair]
                    print("%s had a value, but %s had none" % (pair, inv_pair))

        return dist_values

    def calculate_int_repr(self, normalize=True):
        """
        calculates the representativeness of sounds inside the defined cluster.
        :param normalize: calculate the Normalized PMI if True; (unnormalized) PMI otherwise.
        :return:
        """
        repr_per_combination = {}

        probs = self.int_conf_mat / np.sum(self.int_conf_mat)
        sums = self.int_conf_mat.sum(axis=1)
        marginal_probs = sums / np.sum(sums)

        # zero_index = self.enc_dict["-"]
        # prob_distr = self.get_probabilities(self.trs_encoded_slice)
        # prob_distr[zero_index] = int_row_sums[zero_index] / sum(int_row_sums)
        # REPRESENTATIVENESS is only calculated per symbol
        for i in self.dec_dict.keys():
            prob_i = marginal_probs[i]
            prob_i_j = probs[i, i]

            # if one of the probabilities is zero, do not consider this sound.
            if (prob_i_j * prob_i) != 0:
                repr = math.log(prob_i_j / (prob_i ** 2))

                if normalize:
                    repr = repr / (- math.log(prob_i_j))
                repr_per_combination[self.dec_dict[i]] = repr

        return repr_per_combination

    def calculate_ext_repr(self, normalize=True):
        """
        calculates the representativeness of sounds outside the defined cluster.
        :param normalize: calculate the Normalized PMI if True; (unnormalized) PMI otherwise.
        :return:
        """
        repr_per_combination = {}

        probs = self.ext_conf_mat / np.sum(self.ext_conf_mat)
        sums = self.ext_conf_mat.sum(axis=1)
        marginal_probs = sums / np.sum(sums)

        # REPRESENTATIVENESS is only calculated per symbol
        for i in self.dec_dict.keys():
            prob_i = marginal_probs[i]
            prob_i_j = probs[i, i]

            # if one of the probabilities is zero, do not consider this sound.
            if (prob_i_j * prob_i) != 0:
                repr = math.log(prob_i_j / (prob_i ** 2))

                if normalize:
                    repr = repr / (- math.log(prob_i_j))
                repr_per_combination[self.dec_dict[i]] = repr

        return repr_per_combination

    def calculate_repr(self, normalize=True):
        int_repr = self.calculate_int_repr(normalize=normalize)
        ext_repr = self.calculate_ext_repr(normalize=normalize)

        # define repr as harmonic mean of internal and external repr
        repr = {}
        for i, j in permutations(self.dec_dict.keys(), 2):
            i = self.dec_dict[i]
            j = self.dec_dict[j]
            char_pair = "[%s] : [%s]" % (i, j)
            repr[char_pair] = harmonic_mean(int_repr.get(i, 0), ext_repr.get(j, 0))

        return repr

    def calculate_metrics(self, normalize=True):
        repr = self.calculate_repr(normalize=normalize)
        dist = self.calculate_dist(normalize=normalize)

        characteristicness = {pair: harmonic_mean(dist[pair], repr[pair]) for pair in dist}

        return characteristicness, repr, dist

    def get_frequencies(self):
        return self.frequencies


def harmonic_mean(dist, repr):
    if dist == 0.0 and repr == 0.0:
        return 0.0

    return 2 * ((dist * repr) / (dist + repr))
