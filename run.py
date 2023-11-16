from shibboleth.calculate import ShibbolethCalculator
from shibboleth.utils.io import write_results_to_file, read_cluster_file
from os import path
from glob import glob


"""
- form 46087 contains morphological variation and should be ignored -- but how?
- check the -[V]lo suffixes
"""


clusters_dir = path.join("resources/site_clusters")
results_dir = path.join("resources/results")
data_fp = path.join("resources/data/alt.tsv")

for cluster_file in glob("resources/site_clusters/*.txt"):
    # read in defined clusters
    sites = read_cluster_file(path.join(cluster_file))
    cluster_name = cluster_file.split("/")[-1].replace(".txt", "")

    if not cluster_name.startswith("gianelli"):
        continue

    # calculate metrics
    sc = ShibbolethCalculator(sites, data_fp, skip_sites=["225_italiano"], realign=False)
    sc.count_phonetic_correspondences()
    charac, repr, dist = sc.calculate_metrics(normalize=True)
    freq = sc.get_frequencies()

    # write metrics to file
    out_fp = path.join(f"resources/results/all/{cluster_name}.txt")
    write_results_to_file(out_fp, charac, repr, dist, freq)

    # filter out infrequent sounds and write filtered results to file
    out_fp_filtered = path.join(f"resources/results/filtered/{cluster_name}.txt")
    write_results_to_file(out_fp_filtered, charac, repr, dist, sc.get_frequencies(), threshold=50)
