# Generating the data

The file `alt.tsv` represents a condensed version of the full [Lexibank dataset](https://github.com/lexibank/alt) derived from the Atlante Lessicale Toscano. The TSV file, which is reduced to the essential information required for this study, was generated with the `edictor wordlist` shell command from the `pyedictor` package.

The precise commands are given in the `Makefile`. To re-generate the `alt.tsv` file from the Lexibank data, you just need to run:

```bash
$ make
```

This command clones the Lexibank dataset, installs `pyedictor` and runs the according script. To then remove the Lexibank dataset from your disk, you can run:

```bash
$ make clear
```
