import numpy as np
from pycldf import Dataset


def read_areas_from_file(filename):
    """
    a helper method to conveniently read cluster/area annotations for sites in dataset.
    both sites and areas need to be represented as integers.
    :param filename: the path to the file to read from
    :return: a dictionary containing the information which site belongs to which cluster
    """
    areas_from_file = {}

    with open(filename) as f:
        for line in f:
            if not line:
                continue
            site, area = line.strip().split()
            areas_from_file[int(site)] = int(area)

    return areas_from_file


def read_pmi_scores(fp):
    """
    reads previously inferred PMI scores for sound pairs from a tab-separated file,
    where the first two columns represent the sound pair and the third column contains the score.
    :param fp: the path to the file to read from
    :return: a tuple of an encoding dictionary (sound -> int), a decoding dictionary (int -> sound),
                and the scores represented as a 2d numpy matrix.
    """
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


def load_cldf_data(cldf_meta_fp):
    data = Dataset.from_metadata(cldf_meta_fp)
    forms = list(data["FormTable"])
    params = list(data["ParameterTable"])
    sites = list(data["LanguageTable"])

    return forms, params, sites


def write_results_to_file(fp, charac, repr, dist, frequencies, threshold=0):
    """
    write the results to a given file, sorted by characteristicness in descending order.
    :param fp: the path of the output file
    :param charac: the dictionary where characteristicness values are stored
    :param repr: the dictionary where representativeness values are stored
    :param dist: the dictionary where distinctiveness values are stored
    :param frequencies: a dictionary indicating global token frequencies
    :param threshold: a custom threshold to filter out correspondences with sounds that don't exceed it
    :return:
    """
    sorted_keys = sorted(charac, key=charac.get, reverse=True)
    idio_sorted = {}
    for k in sorted_keys:
        idio_sorted[k] = charac[k]

    with open(fp, "w") as f:
        f.write("CORR\tDIST\tREPR\tCHAR\tFREQ\n")
        for pair in idio_sorted:
            sound1, sound2 = pair.split(" : ")
            sound1 = sound1[1:-1]
            sound2 = sound2[1:-1]
            freq1 = frequencies[sound1] if sound1 in frequencies else 0
            freq2 = frequencies[sound2] if sound2 in frequencies else 0
            pair_dist = dist[pair]
            pair_repr = repr[pair]
            pair_idio = idio_sorted[pair]
            if (freq1 > threshold or sound1 == "-") and (freq2 > threshold or sound2 == "-"):
                f.write(
                    "%s\t%.3f\t%.3f\t%.3f\t%i|%i\n" % (pair, pair_dist, pair_repr, pair_idio, freq1, freq2))

