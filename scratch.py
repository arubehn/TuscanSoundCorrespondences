from lingpy import *
from collections import defaultdict
import pandas as pd

"""
data = Wordlist.from_cldf("resources/data/ALT/cldf/Wordlist-metadata.json")

segments = defaultdict(list)

for row in data.iter_rows(*data.columns):
    concept_id = row[1]
    segments_for_concept = row[7]
    if segments_for_concept not in segments[concept_id]:
        segments[concept_id].append(segments_for_concept)


for cogset in segments.values():
    msa = Multiple(cogset)
    msa.align('progressive')

    print(msa)
    print("\n" + 100 * "=" + "\n")
"""


df = pd.read_csv("resources/data/ALT/raw/ALT-standardized_forms.csv", header=0, index_col=0)

print(df.loc["fragola"]["207 Talamone"])
