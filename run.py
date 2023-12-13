from shibboleth.calculate import ShibbolethCalculator
from shibboleth.utils.io import write_results_to_file, read_cluster_file
from shibboleth.utils.metrics import sound_metrics, write_metrics_to_file
from lingpy import Wordlist
from os import path
from glob import glob


"""
- form 46087 contains morphological variation and should be ignored -- but how?
- check the -[V]lo suffixes
"""


clusters_dir = path.join("resources/site_clusters")
results_dir = path.join("resources/results")
data_fp = path.join("resources/data/alt.tsv")

# calculate sound metrics and save them to a file
wl = Wordlist(data_fp)
metrics = sound_metrics(wl, skip_sites=["225_italiano"])
write_metrics_to_file(path.join("resources/data/sound_metrics.tsv"), metrics)

for cluster_file in glob("resources/site_clusters/*.txt"):
    # read in defined clusters
    sites = read_cluster_file(path.join(cluster_file))
    cluster_name = cluster_file.split("/")[-1].replace(".txt", "")

    # calculate metrics
    sc = ShibbolethCalculator(sites, data_fp, skip_sites=["225_italiano"], realign=False)
    charac, repr, dist = sc.calculate_metrics(normalize=True)
    freq = sc.get_frequencies()

    # write metrics to file
    out_fp = path.join(f"resources/results/all/{cluster_name}.tsv")
    write_results_to_file(out_fp, charac, repr, dist, freq)

    # filter out infrequent sounds and write filtered results to file
    out_fp_filtered = path.join(f"resources/results/filtered/{cluster_name}.tsv")
    write_results_to_file(out_fp_filtered, charac, repr, dist, sc.get_frequencies(), threshold=50)


# compare cluster 2 to 3/4, excluding 1.
sites = read_cluster_file(path.join("resources/site_clusters/gianelli_savoia_3_4.txt"))
excluded = read_cluster_file("resources/site_clusters/gianelli_savoia_1.txt")
excluded.append("225_italiano")

# calculate metrics
sc = ShibbolethCalculator(sites, data_fp, skip_sites=excluded, realign=False)
charac, repr, dist = sc.calculate_metrics(normalize=True)
freq = sc.get_frequencies()

# write metrics to file
out_fp = path.join(f"resources/results/all/gianelli_savoia_3_vs_2.tsv")
write_results_to_file(out_fp, charac, repr, dist, freq)

# filter out infrequent sounds and write filtered results to file
out_fp_filtered = path.join(f"resources/results/filtered/gianelli_savoia_3_vs_2.tsv")
write_results_to_file(out_fp_filtered, charac, repr, dist, sc.get_frequencies(), threshold=50)
