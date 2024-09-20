from collections import defaultdict


def sound_metrics(wl, skip_concepts=None, skip_sites=None, skip_forms=None):
    if skip_concepts is None:
        skip_concepts = []

    if skip_sites is None:
        skip_sites = []

    if skip_forms is None:
        skip_forms = []

    sound_to_sites = defaultdict(set)
    sound_to_concept = defaultdict(set)
    sound_frequencies = defaultdict(int)

    for row in wl.iter_rows("doculect", "concept", "tokens"):
        form_id, site, concept, tokens = row
        if not (form_id in skip_forms or site in skip_sites or concept in skip_concepts):
            for sound in tokens:
                sound_to_sites[sound].add(site)
                sound_to_concept[sound].add(concept)
                sound_frequencies[sound] += 1

    # aggregate that information in a single, nested dictionary
    metrics = defaultdict(lambda: defaultdict(int))

    for sound, sites in sound_to_sites.items():
        metrics[sound]["site_coverage"] = len(sites)

    for sound, concepts in sound_to_concept.items():
        metrics[sound]["concept_coverage"] = len(concepts)

    for sound, freq in sound_frequencies.items():
        metrics[sound]["frequency"] = freq

    return metrics


def write_metrics_to_file(fp, metrics, delimiter="\t"):
    with open(fp, "w") as f:
        f.write(delimiter.join(["SOUND", "FREQUENCY", "CONCEPT_COVERAGE", "SITE_COVERAGE"]) + "\n")
        for sound in metrics.keys():
            frequency = str(metrics[sound]["frequency"])
            concept_coverage = str(metrics[sound]["concept_coverage"])
            site_coverage = str(metrics[sound]["site_coverage"])
            f.write(delimiter.join([sound, frequency, concept_coverage, site_coverage]) + "\n")
