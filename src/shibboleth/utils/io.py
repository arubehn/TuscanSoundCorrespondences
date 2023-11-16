from lingpy import Wordlist
from os import path


def load_data(data_fp):
    data_fp = path.join(data_fp)

    try:
        data = Wordlist(data_fp)
    except:
        data = Wordlist.from_cldf(data_fp)

    return data


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
            if not line.strip():
                continue
            site, area = line.strip().split()
            areas_from_file[int(site)] = int(area)

    return areas_from_file


def read_cluster_file(filename):
    """
    reads in sites for a user-defined cluster stored in a comma-separated file.
    :param filename: the path to the file to read from
    :return: a list of varieties defined to be in the cluster
    """
    with open(filename) as f:
        return f.read().split(",")


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

