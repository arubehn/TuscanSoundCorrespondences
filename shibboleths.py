import pandas as pd
import numpy as np
import re
from itertools import combinations, product, permutations
from nw import nw
import math
from pycldf import Dataset

areas = {36: 2,
         37: 3,
         38: 2,
         39: 3,
         40: 2,
         41: 2,
         7: 3,
         11: 3,
         12: 3,
         13: 3,
         14: 3,
         15: 3,
         16: 3,
         17: 3,
         18: 3,
         19: 3,
         20: 2,
         21: 2,
         22: 2,
         23: 2,
         24: 1,
         25: 1,
         26: 1,
         27: 1,
         28: 1,
         29: 1,
         30: 1,
         33: 3,
         34: 2,
         35: 2,
         42: 2,
         43: 2,
         44: 2,
         45: 2,
         46: 1,
         47: 1,
         48: 1,
         49: 1,
         50: 1,
         51: 1,
         52: 1,
         53: 1,
         54: 1,
         55: 1,
         56: 1,
         57: 1,
         58: 1,
         59: 1,
         60: 1,
         61: 1,
         62: 1,
         63: 1,
         64: 3,
         65: 3,
         66: 3,
         67: 3,
         68: 2,
         69: 2,
         70: 2,
         71: 2,
         72: 2,
         73: 2,
         74: 2,
         75: 2,
         76: 2,
         77: 1,
         78: 1,
         79: 1,
         80: 1,
         81: 1,
         82: 1,
         83: 1,
         84: 1,
         85: 1,
         86: 1,
         87: 1,
         88: 1,
         89: 1,
         90: 1,
         91: 1,
         92: 3,
         93: 1,
         94: 4,
         95: 3,
         96: 3,
         97: 4,
         98: 4,
         99: 3,
         101: 3,
         102: 3,
         103: 4,
         104: 3,
         105: 3,
         106: 2,
         107: 2,
         108: 2,
         109: 2,
         110: 2,
         111: 2,
         112: 2,
         113: 2,
         114: 2,
         115: 2,
         116: 1,
         117: 2,
         118: 1,
         119: 1,
         120: 1,
         121: 1,
         122: 1,
         123: 1,
         124: 1,
         125: 1,
         126: 1,
         127: 1,
         128: 1,
         129: 1,
         130: 1,
         131: 1,
         132: 1,
         133: 4,
         134: 4,
         135: 4,
         136: 3,
         137: 3,
         138: 3,
         139: 3,
         140: 3,
         141: 3,
         142: 2,
         143: 2,
         144: 2,
         145: 2,
         146: 2,
         147: 2,
         148: 2,
         149: 2,
         150: 2,
         151: 2,
         152: 2,
         153: 2,
         154: 2,
         155: 2,
         156: 2,
         157: 1,
         158: 2,
         159: 1,
         160: 1,
         161: 1,
         162: 2,
         163: 1,
         164: 1,
         165: 3,
         166: 3,
         167: 3,
         168: 3,
         169: 3,
         170: 3,
         171: 3,
         172: 3,
         173: 3,
         174: 3,
         175: 3,
         176: 3,
         177: 3,
         178: 3,
         179: 3,
         180: 3,
         181: 2,
         182: 2,
         183: 2,
         184: 2,
         185: 2,
         186: 2,
         187: 2,
         188: 2,
         189: 2,
         190: 2,
         191: 2,
         192: 2,
         193: 3,
         194: 3,
         195: 3,
         196: 3,
         197: 3,
         198: 3,
         199: 3,
         200: 3,
         201: 3,
         202: 3,
         203: 3,
         204: 3,
         205: 2,
         206: 4,
         207: 3,
         208: 2,
         209: 4,
         210: 3,
         211: 4,
         212: 4,
         213: 4,
         214: 3,
         215: 3,
         216: 3,
         217: 3,
         218: 3,
         219: 3,
         220: 3,
         221: 3,
         222: 3,
         223: 3,
         224: 3
         }


class ShibbolethCalculator(object):
    def __init__(self, varieties, cldf_meta_fp, sim_mat_fp):
        self.varieties = varieties

        self.data = Dataset.from_metadata(cldf_meta_fp)
        self.forms = list(self.data["FormTable"])
        self.params = list(self.data["ParameterTable"])
        self.sites = list(self.data["LanguageTable"])

        self.concepts_enc_dict = {p["Name"]: p["ID"] for p in self.params}
        self.concepts_dec_dict = {p["ID"]: p["Name"] for p in self.params}

        self.sites_enc_dict = {s["Name"]: s["ID"] for s in self.sites}
        self.sites_dec_dict = {s["ID"]: s["Name"] for s in self.sites}

        self.enc_dict, self.dec_dict, self.sim_mat = self.read_pmi_scores(sim_mat_fp)

        self.trs = pd.DataFrame(index=list(self.sites_dec_dict.keys()), columns=list(self.concepts_dec_dict.keys()))
        self.trs_encoded = self.trs.copy()

        for f in self.forms:
            site_id = f["Language_ID"]
            concept_id = f["Parameter_ID"]
            segments = f["Segments"]
            self.trs.loc[site_id, concept_id] = segments
            self.trs_encoded.loc[site_id, concept_id] = [self.enc_dict[x] for x in segments]

        self.int_conf_mat = np.zeros(shape=(len(self.enc_dict), len(self.enc_dict)), dtype=int)
        self.ext_conf_mat = np.zeros(shape=(len(self.enc_dict), len(self.enc_dict)), dtype=int)
        """# TODO change back
        self.int_conf_mat = np.random.randint(100, size=(len(self.enc_dict), len(self.enc_dict)))
        self.ext_conf_mat = np.random.randint(100, size=(len(self.enc_dict), len(self.enc_dict)))"""

        self.trs_slice = self.trs[self.trs.index.astype(int).isin(self.varieties)]
        self.trs_encoded_slice = self.trs_encoded[self.trs_encoded.index.astype(int).isin(self.varieties)]

        self.comp_var_ids = [int(var["ID"]) for var in self.sites if int(var["ID"]) not in self.varieties]
        self.trs_rest = self.trs[self.trs.index.astype(int).isin(self.comp_var_ids)]
        self.trs_encoded_rest = self.trs_encoded[self.trs_encoded.index.astype(int).isin(self.comp_var_ids)]

    def read_pmi_scores(self, fp):
        enc_dict = {}
        dec_dict = {}

        scores_dict = {}

        with open(fp) as f:
            for line in f:
                fields = line.strip().split()
                if fields[0] == "#" or fields[1] == "#":
                    continue
                s1 = fields[0]
                s2 = fields[1]
                score = float(fields[2])
                if s1 not in enc_dict:
                    idx = len(enc_dict)
                    enc_dict[s1] = idx
                    dec_dict[idx] = s1
                if s2 not in enc_dict:
                    idx = len(enc_dict)
                    enc_dict[s2] = idx
                    dec_dict[idx] = s2
                scores_dict[(s1, s2)] = score

        score_matrix = np.zeros((len(enc_dict), len(enc_dict)))
        for pair, score in scores_dict.items():
            s1, s2 = pair
            score_matrix[enc_dict[s1], enc_dict[s2]] = score

        return enc_dict, dec_dict, score_matrix

    def print_rows(self):
        vars_full_names = [self.sites_dec_dict[var] for var in self.varieties]
        print(self.trs[self.trs.index.isin(vars_full_names)])

    def align(self):
        # internal alignments
        for _, data in self.trs_encoded_slice.iteritems():
            values = data.values
            for i, j in permutations(values, 2):
                if not isinstance(i, float) and not isinstance(j, float):
                    alignment = nw(i, j, self.sim_mat, self.enc_dict["-"])
                    alignment_i = alignment[0]
                    alignment_j = alignment[1]
                    for idx in range(len(alignment_i)):
                        self.int_conf_mat[alignment_i[idx], alignment_j[idx]] += 1

        # external alignments
        for index in range(len(self.concepts_enc_dict)):
            int_values = self.trs_encoded_slice.iloc[:, index].values
            ext_values = self.trs_encoded_rest.iloc[:, index].values
            for i, j in product(int_values, ext_values):
                if not isinstance(i, float) and not isinstance(j, float):
                    alignment = nw(i, j, self.sim_mat, self.enc_dict["-"])
                    alignment_i = alignment[0]
                    alignment_j = alignment[1]
                    for idx in range(len(alignment_i)):
                        self.ext_conf_mat[alignment_i[idx], alignment_j[idx]] += 1

    def get_probabilities(self, df, invert_axis=False):
        occurrences_per_char = {}
        for i in range(len(self.enc_dict)):
            occurrences_per_char[i] = 0
        sum = 0

        if invert_axis:
            for _, col in df.iteritems():  # iterate over rows
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
        dist_per_combination = {}
        ext_col_sums = self.ext_conf_mat.sum(axis=0)
        ext_row_sums = self.ext_conf_mat.sum(axis=1)
        zero_index = self.enc_dict["-"]
        int_prob_distr = self.get_probabilities(self.trs_encoded_slice)
        int_prob_distr[zero_index] = ext_row_sums[zero_index] / sum(ext_row_sums)
        ext_prob_distr = self.get_probabilities(self.trs_encoded_rest)
        ext_prob_distr[zero_index] = ext_col_sums[zero_index] / sum(ext_col_sums)
        # ASSUMPTION: ext [i] changes to internal [j]
        for i, j in permutations(self.dec_dict.keys(), 2):
            char_pair = "[%s] -> [%s]" % (self.dec_dict[i], self.dec_dict[j])
            prob_i = ext_prob_distr[i]
            prob_j = int_prob_distr[j]
            prob_i_j = self.ext_conf_mat[j, i] / sum(ext_row_sums)
            try:
                dist = math.log(prob_i_j / (prob_i * prob_j))
                if not math.isnan(dist):
                    if normalize:
                        n_dist = dist / (- math.log2(prob_i_j))
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
            inv_pair = "[%s] -> [%s]" % (char2, char1)
            if dist_per_combination[pair] > 0:
                if inv_pair in dist_per_combination:
                    dist_new = dist_per_combination[pair] - dist_per_combination[inv_pair]
                    dist_values[pair] = dist_new
                else:
                    dist_values[pair] = dist_per_combination[pair]
                    print("%s had a value, but %s had none" % (pair, inv_pair))

        return dist_values

    def calculate_int_repr(self, normalize=False):
        repr_per_combination = {}
        int_row_sums = self.int_conf_mat.sum(axis=1)
        zero_index = self.enc_dict["-"]
        prob_distr = self.get_probabilities(self.trs_encoded_slice)
        prob_distr[zero_index] = int_row_sums[zero_index] / sum(int_row_sums)
        # REPRESENTATIVENESS is only calculated per symbol
        for i in self.dec_dict.keys():
            prob_i = prob_distr[i]
            prob_i_j = self.int_conf_mat[i, i] / sum(int_row_sums)
            try:
                repr = math.log(prob_i_j / (prob_i ** 2))
                if math.isnan(repr):
                    repr_per_combination[self.dec_dict[i]] = 0
                else:
                    if normalize:
                        n_repr = repr / (- math.log2(prob_i_j))
                        repr_per_combination[self.dec_dict[i]] = n_repr
                    else:
                        repr_per_combination[self.dec_dict[i]] = repr
            except:
                repr_per_combination[self.dec_dict[i]] = 0

        return repr_per_combination

    def calculate_ext_repr(self, normalize=False):
        repr_per_combination = {}
        ext_row_sums = self.ext_conf_mat.sum(axis=1)
        zero_index = self.enc_dict["-"]
        prob_distr = self.get_probabilities(self.trs_encoded_slice)
        prob_distr[zero_index] = ext_row_sums[zero_index] / sum(ext_row_sums)
        # REPRESENTATIVENESS is only calculated per symbol
        for i in self.dec_dict.keys():
            prob_i = prob_distr[i]
            prob_i_j = self.int_conf_mat[i, i] / sum(ext_row_sums)
            try:
                repr = math.log(prob_i_j / (prob_i ** 2))
                if math.isnan(repr):
                    repr_per_combination[self.dec_dict[i]] = 0
                else:
                    if normalize:
                        n_repr = repr / (- math.log2(prob_i_j))
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
            char_pair = "[%s] -> [%s]" % (i, j)
            repr[char_pair] = self.harmonic_mean(int_repr[i], ext_repr[j])

        return repr

    def harmonic_mean(self, dist, repr):
        try:
            return 2 * ((dist * repr) / (dist + repr))
        except:
            return 0


if __name__ == "__main__":
    samples = ["elba", "general_gorgia", "gianelli_savoia_1", "gianelli_savoia_2", "gianelli_savoia_3",
              "gianelli_savoia_4", "gianelli_savoia_3_4", "pisa_livorno"]
    samples += ["b_B", "d_D", "general_gorgia_contextfree"]
    #samples = ["gianelli_savoia_1_2"]

    for sample in samples:
        with open("./ALT/site_clusters/%s.txt" % sample, "r") as f:
            v = f.read().split(",")

        v = [int(num) for num in v]
        # v = [102, 103, 104]

        t = ShibbolethCalculator(v, "./ALT/cldf/Wordlist-metadata.json", "./ALT/PMI_scores.tsv")
        print("Initialized tables.")
        t.align()
        # print("Finished alignments.")
        dist = t.calculate_dist(normalize=True)
        repr = t.calculate_repr(normalize=True)
        idio = {pair: t.harmonic_mean(dist[pair], repr[pair]) for pair in dist}

        sorted_keys = sorted(idio, key=idio.get, reverse=True)
        idio_sorted = {}
        for k in sorted_keys:
            idio_sorted[k] = idio[k]

        with open("results/%s.txt" % sample, "w") as f:
            f.write("CORR\tDIST\tREPR\tIDIO\n")
            for pair in idio_sorted:
                pair_dist = dist[pair]
                pair_repr = repr[pair]
                pair_idio = idio_sorted[pair]
                f.write("%s\t%.3f\t%.3f\t%.3f\n" % (pair, pair_dist, pair_repr, pair_idio))
