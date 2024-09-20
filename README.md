# TuscanSoundCorrespondences

This repository contains all the code and data accompanying our paper:
> Rubehn, A., Montemagni, S., & Nerbonne, J. (2024). Extracting Tuscan phonetic correspondences from dialect pronunciations automatically. *Language Dynamics and Change*, 14(1), 1-33. https://doi.org/10.1163/22105832-bja10034

If you (re-)use parts of this code, please cite our work accordingly.

## Installation

Clone into this repository on your machine:

```bash
$ git clone https://github.com/arubehn/TuscanSoundCorrespondences
$ cd TuscanSoundCorrespondences
```

It is highly recommended to [set up a fresh virtual environment](https://docs.python.org/3/library/venv.html).

In order to set up the project and install all required dependencies, simply run

```bash
$ pip install -e .
```

## Replication

Once the project is set up correctly, all analyses discussed in the paper can be replicated by executing the `run.py` script.

```bash
$ python3 run.py
```

## Analyzing new data

You can analyze your own data with just a few lines of code -- you just need to define the varieties of interest (*V*) and your data filepath. Input data is expected to be present either as a [CLDF repository](https://github.com/cldf/cldf), or as a TSV file with headers according to the [EDICTOR](https://edictor.org) specifications.

```python
from shibboleth.calculate import ShibbolethCalculator

data = "YOUR/DATA/FILE/PATH"
varieties = ["your", "set", "of", "varieties"]  # needs to match the language id given in the data exactly

sc = ShibbolethCalculator(varieties, data)
charac, repr, dist = sc.calculate_metrics(normalize=True) # normalize=True calculates normalized PMI; otherwise plain PMI is calculated
```

If your data already Multiple Sequence Alignments (MSA), those are used by default. Otherwise, alignments are generated using the SCA algorithm implemented in [LingPy](https://github.com/lingpy/lingpy).

## Dataset

The dataset used in this study is derived from the Atlante Lessicale Toscano (ALT). Our derived dataset was published within the Lexibank collection under https://github.com/lexibank/alt/.
