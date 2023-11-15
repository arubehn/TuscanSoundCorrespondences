from shibboleth.calculate import ShibbolethCalculator
from shibboleth.utils.io import write_results_to_file
from os import path


"""
- form 46087 contains morphological variation and should be ignored -- but how?
- check the -[V]lo suffixes
"""

# test run
with open("resources/site_clusters/gianelli_savoia_1.txt") as f:
    sites = f.read().split(",")

data_fp = path.join("resources/data/alt.tsv")

sc = ShibbolethCalculator(sites, data_fp, skip_sites=["225_italiano"])
sc.count_phonetic_correspondences()
charac, repr, dist = sc.calculate_metrics(normalize=True)

out_fp = path.join("resources/results/TEST_NEW_1.txt")
write_results_to_file(out_fp, charac, repr, dist, sc.get_frequencies())

out_fp_filtered = path.join("resources/results/TEST_NEW_1_FILTERED.txt")
write_results_to_file(out_fp_filtered, charac, repr, dist, sc.get_frequencies(), threshold=50)
