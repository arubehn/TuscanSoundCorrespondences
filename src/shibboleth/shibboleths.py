import pandas as pd
import numpy as np
import re
from itertools import product, permutations
from shibboleth.utils.nw import nw
import math
from shibboleth.utils.io import *


# TODO re-integrate function for caching confusion matrices
class ShibbolethCalculator(object):
    def __init__(self, varieties, cldf_meta_fp, sim_mat_fp):
        """
        initialize the calculator class; defining the cluster of varieties in question,
        loading in the lexical data, and providing a similarity matrix over all sounds for constructing alignments.
        :param varieties: the defined cluster of varieties to be investigated.
        :param cldf_meta_fp: the filepath to the CLDF metadata file corresponding to the dataset
        :param sim_mat_fp: the filepath to retrieve the similarity matrix for sound pairs from
        """
        # store the defined cluster of varieties
        self.varieties = varieties

        # load lexical data
        self.forms, self.params, self.sites = load_cldf_data(cldf_meta_fp)

        # set up encoding and decoding dictionaries for parameter (=concept) IDs
        self.concepts_enc_dict = {p["Name"]: p["ID"] for p in self.params}
        self.concepts_dec_dict = {p["ID"]: p["Name"] for p in self.params}

        # set up encoding and decoding dictionaries for site (=variety) IDs
        self.sites_enc_dict = {s["Name"]: s["ID"] for s in self.sites}
        self.sites_dec_dict = {s["ID"]: s["Name"] for s in self.sites}

        # load sound similarity matrix (quantified by PMI scores)
        self.enc_dict, self.dec_dict, self.sim_mat = read_pmi_scores(sim_mat_fp)

        # set up a matrix that will be populated with all forms,
        # where the rows represent the sites and the columns indicate the concepts
        self.form_matrix = pd.DataFrame(index=list(self.sites_dec_dict.keys()), columns=list(self.concepts_dec_dict.keys()))

        # set up another form matrix where all segments are represented as their integer ID
        self.form_matrix_encoded = self.form_matrix.copy()

        # keep track of the global frequencies of each sound
        self.frequencies = {}

        # populate form matrices
        for f in self.forms:
            site_id = f["Language_ID"]
            concept_id = f["Parameter_ID"]
            segments = f["Segments"]
            for seg in segments:
                if seg in self.frequencies:
                    self.frequencies[seg] += 1
                else:
                    self.frequencies[seg] = 1
            self.form_matrix.loc[site_id, concept_id] = segments
            self.form_matrix_encoded.loc[site_id, concept_id] = [self.enc_dict[x] for x in segments if x != "_"]

        # initialize 3 confusion matrices:
        # one for inside the given cluster (int), one for all varieties outside the cluster (ext),
        # one for alignments between varieties from inside and outside the cluster (cross)
        self.int_conf_mat = np.zeros(shape=(len(self.enc_dict), len(self.enc_dict)), dtype=int)
        self.ext_conf_mat = np.zeros(shape=(len(self.enc_dict), len(self.enc_dict)), dtype=int)
        self.cross_conf_mat = np.zeros(shape=(len(self.enc_dict), len(self.enc_dict)), dtype=int)

        # retrieve partial form matrix for the defined cluster
        self.int_form_matrix = self.form_matrix[self.form_matrix.index.astype(int).isin(self.varieties)]
        self.int_form_matrix_encoded = self.form_matrix_encoded[self.form_matrix_encoded.index.astype(int).isin(self.varieties)]

        # retrieve partial form matrix for all sites outside the defined cluster
        self.comp_var_ids = [int(var["ID"]) for var in self.sites if int(var["ID"]) not in self.varieties]
        self.ext_form_matrix = self.form_matrix[self.form_matrix.index.astype(int).isin(self.comp_var_ids)]
        self.ext_form_matrix_encoded = self.form_matrix_encoded[self.form_matrix_encoded.index.astype(int).isin(self.comp_var_ids)]

    def print_rows(self):
        """
        convenience method for debugging; prints all the forms pertaining to the defined site cluster
        :return:
        """
        vars_full_names = [self.sites_dec_dict[var] for var in self.varieties]
        print(self.form_matrix[self.form_matrix.index.isin(vars_full_names)])

    def align(self):
        """
        construct pairwise alignments for all pairs in (int x ext), (int x int), (ext x ext)
        under the same concept, where int and ext refer to the sets of varieties inside and outside
        the defined clusters.
        """
        # internal alignments - including self alignments
        for _, data in self.int_form_matrix_encoded.iteritems():
            values = data.values
            for val in values:  # self alignment
                if isinstance(val, float):
                    continue
                for i in val:
                    self.int_conf_mat[i, i] += 1
            for i, j in permutations(values, 2):  # pairwise alignments
                if not isinstance(i, float) and not isinstance(j, float):
                    alignment = nw(i, j, self.sim_mat, self.enc_dict["-"])
                    alignment_i = alignment[0]
                    alignment_j = alignment[1]
                    for idx in range(len(alignment_i)):
                        self.int_conf_mat[alignment_i[idx], alignment_j[idx]] += 1
                        self.int_conf_mat[alignment_j[idx], alignment_i[idx]] += 1

        # external alignments
        for _, data in self.ext_form_matrix_encoded.iteritems():
            values = data.values
            for val in values:  # self alignment
                if isinstance(val, float):
                    continue
                for i in val:
                    self.ext_conf_mat[i, i] += 1
            for i, j in permutations(values, 2):
                if not isinstance(i, float) and not isinstance(j, float):
                    alignment = nw(i, j, self.sim_mat, self.enc_dict["-"])
                    alignment_i = alignment[0]
                    alignment_j = alignment[1]
                    for idx in range(len(alignment_i)):
                        self.ext_conf_mat[alignment_i[idx], alignment_j[idx]] += 1
                        self.ext_conf_mat[alignment_j[idx], alignment_i[idx]] += 1

        # cross-cluster alignments
        for index in range(len(self.concepts_enc_dict)):
            int_values = self.int_form_matrix_encoded.iloc[:, index].values
            ext_values = self.ext_form_matrix_encoded.iloc[:, index].values
            for i, j in product(int_values, ext_values):
                if not isinstance(i, float) and not isinstance(j, float):
                    alignment = nw(i, j, self.sim_mat, self.enc_dict["-"])
                    alignment_i = alignment[0]
                    alignment_j = alignment[1]
                    for idx in range(len(alignment_i)):
                        self.cross_conf_mat[alignment_i[idx], alignment_j[idx]] += 1

    def get_probabilities(self, df, invert_axis=False):
        """
        calculate conditional probabilities for rows or columns in a dataframe.
        :param df: a 2d numpy matrix containing the raw counts
        :param invert_axis: if True, calculate probabilities over columns; over rows otherwise.
        :return:
        """
        occurrences_per_char = {}
        for i in range(len(self.enc_dict)):
            occurrences_per_char[i] = 0
        sum = 0

        if invert_axis:
            for _, col in df.iteritems():  # iterate over columns
                for _, value in col.items():
                    if not isinstance(value, float):
                        for char in value:
                            sum += 1
                            if char in occurrences_per_char:
                                occurrences_per_char[char] += 1
                            else:
                                occurrences_per_char[char] = 1
        else:
            for _, row in df.iterrows():  # iterate over rows
                for _, value in row.items():
                    if not isinstance(value, float):
                        for char in value:
                            sum += 1
                            if char in occurrences_per_char:
                                occurrences_per_char[char] += 1
                            else:
                                occurrences_per_char[char] = 1

        probabilities_per_char = {}
        for char in occurrences_per_char:
            probabilities_per_char[char] = occurrences_per_char[char] / sum

        return probabilities_per_char

    def calculate_dist(self, normalize=False):
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

        print(np.sum(probs))
        print(np.sum(marginal_probs_cols))
        print(np.sum(marginal_probs_rows))

        #zero_index = self.enc_dict["-"]
        #int_prob_distr = self.get_probabilities(self.trs_encoded_slice)
        #int_prob_distr[zero_index] = ext_row_sums[zero_index] / sum(ext_row_sums)
        #ext_prob_distr = self.get_probabilities(self.trs_encoded_rest)
        #ext_prob_distr[zero_index] = ext_col_sums[zero_index] / sum(ext_col_sums)
        # ASSUMPTION: internal [i] corresponds to external [j]
        for i, j in permutations(self.dec_dict.keys(), 2):
            char_pair = "[%s] : [%s]" % (self.dec_dict[i], self.dec_dict[j])
            if char_pair == "[-] : [g]":
                print("breakpoint")
            prob_i = marginal_probs_rows[i]
            prob_j = marginal_probs_cols[j]
            prob_i_j = probs[i, j]
            try:
                dist = math.log(prob_i_j / (prob_i * prob_j))
                if not math.isnan(dist):
                    if normalize:
                        n_dist = dist / (- math.log(prob_i_j))
                        dist_per_combination[char_pair] = n_dist
                    else:
                        dist_per_combination[char_pair] = dist
                # npmi = pmi / (- math.log2(prob_i_j))
                # if npmi <= 1:
                #    pmi_per_combination[char_pair] = npmi
            except:
                continue

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

    def calculate_int_repr(self, normalize=False):
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
            try:
                repr = math.log(prob_i_j / (prob_i ** 2))
                if math.isnan(repr):
                    repr_per_combination[self.dec_dict[i]] = 0
                else:
                    if normalize:
                        n_repr = repr / (- math.log(prob_i_j))
                        repr_per_combination[self.dec_dict[i]] = n_repr
                    else:
                        repr_per_combination[self.dec_dict[i]] = repr
            except:
                repr_per_combination[self.dec_dict[i]] = 0

        return repr_per_combination

    def calculate_ext_repr(self, normalize=False):
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
            try:
                repr = math.log(prob_i_j / (prob_i ** 2))
                if math.isnan(repr):
                    repr_per_combination[self.dec_dict[i]] = 0
                else:
                    if normalize:
                        n_repr = repr / (- math.log(prob_i_j))
                        repr_per_combination[self.dec_dict[i]] = n_repr
                    else:
                        repr_per_combination[self.dec_dict[i]] = repr
            except:
                repr_per_combination[self.dec_dict[i]] = 0

        return repr_per_combination

    def calculate_repr(self, normalize=False):
        int_repr = self.calculate_int_repr(normalize=normalize)
        ext_repr = self.calculate_ext_repr(normalize=normalize)

        # define repr as harmonic mean of internal and external repr
        repr = {}
        for i, j in permutations(self.dec_dict.keys(), 2):
            i = self.dec_dict[i]
            j = self.dec_dict[j]
            char_pair = "[%s] : [%s]" % (i, j)
            if i == "-":
                repr[char_pair] = ext_repr[j]
            elif j == "-":
                repr[char_pair] = int_repr[i]
            else:
                repr[char_pair] = harmonic_mean(int_repr[i], ext_repr[j])

        return repr

    def calculate_metrics(self, normalize=False):
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
