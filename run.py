from shibboleth.shibboleths import ShibbolethCalculator
from shibboleth.utils.io import write_results_to_file
from os import path


# test run
with open("resources/site_clusters/gianelli_savoia_1_2.txt") as f:
    sites = f.read().split(",")

data_fp = path.join("resources/data/alt.tsv")

sc = ShibbolethCalculator(sites, data_fp, skip_sites=["225_italiano"])
sc.count_phonetic_correspondences()
charac, repr, dist = sc.calculate_metrics(normalize=True)

out_fp = path.join("resources/results/TEST_NEW.txt")
write_results_to_file(out_fp, charac, repr, dist, sc.get_frequencies())
