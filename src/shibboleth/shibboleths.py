import pandas as pd
import numpy as np
import re
from itertools import product, permutations, combinations
from shibboleth.utils.nw import nw
import math
from shibboleth.utils.io import *
from lingpy import Wordlist, Multiple, Alignments
from collections import defaultdict


# TODO re-integrate function for caching confusion matrices
class ShibbolethCalculator(object):
    def __init__(self, varieties, data_fp, skip_concepts=None, skip_sites=None):
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
            self.skip_concept = []
        else:
            self.skip_concept = skip_concepts

        if skip_sites is None:
            self.skip_sites = []
        else:
            self.skip_sites = skip_sites

        # load lexical data as wordlist
        self.data = Wordlist(data_fp)

        # store defined variety cluster by given matching strategy
        # TODO enable passing down different matching strategies
        self.varieties = []
        for var_in_data in self.data.cols:
            for given_var in varieties:
                if var_in_data.startswith(given_var):
                    self.varieties.append(var_in_data)
                    continue

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

        """
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
        self.frequencies = defaultdict(int)

        # populate form matrices
        for f in self.forms:
            site_id = f["Language_ID"]
            concept_id = f["Parameter_ID"]
            segments = f["Segments"]
            for seg in segments:
                self.frequencies[seg] += 1
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
        """

    def align(self):
        """
        construct multiple alignments for all cognate sets.
        """
        self.data.renumber("concept", "cogid")
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
        for concept in self.data.rows:
            if concept in self.skip_concept:
                continue
            # get all form IDs corresponding to the concept
            form_ids = self.data.get_list(row=concept, flat=True)
            # retrieve data for all possible form pairs within the concept
            for id1, id2 in combinations(form_ids, 2):
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
