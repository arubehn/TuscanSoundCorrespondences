import re
from pycldf import Wordlist


"""
This preprocessing script converts the raw ALT data to the standardized dataset
according to the CLDF initiative. 
"""


def tokenize(word):
    """
    a simple greedy tokenizer suited for the ALT dataset
    :word: the word to be tokenized
    """
    tokens = re.findall(".[ː̞ʲ̃'̭]*", word)

    assert "".join(tokens) == word

    return tokens


def read_data(fp="../../../ALT/raw/ALT-standardized_forms.csv"):
    """
    read the data from a csv file, return forms, languages and parameters as required by the CLDF standard
    :param fp: the path to the raw csv file
    :return: CLDF-compatible forms, languages, and parameters
    """
    forms, params, langs = [], [], []
    word_id = 10000
    with open(fp) as f:
        for i, line in enumerate(f):
            fields = line.strip().split(",")
            if fields[0] == "":
                langs_list = fields[1:]
                for l in langs_list:
                    tmp = l.split()
                    lang_id = tmp[0]
                    lang_name = "".join(tmp[1:])
                    langs.append({"ID": lang_id, "Name": lang_name})
            else:
                param = fields[0]
                param_id = str(i+1000)
                params.append({"ID": param_id, "Name": param})
                param_forms = fields[1:]
                for lang_idx, form in enumerate(param_forms):
                    if form == "":
                        continue
                    segments = tokenize(form)
                    lang_id = langs[lang_idx]["ID"]
                    forms.append({"ID": str(word_id), "Form": form, "Segments": segments,
                                  "Language_ID": lang_id, "Parameter_ID": param_id})
                    word_id += 1

    return forms, params, langs


def write_cldf(forms, params, langs, output_dir="./ALT/cldf"):
    """
    write data in CLDF format to a given directory.
    :param forms: the FormTable
    :param params: the ParameterTable (i.e. the concepts)
    :param langs: the LanguageTable
    :param output_dir: the output directory
    :return:
    """
    dataset = Wordlist.in_dir(output_dir)
    dataset.add_component("ParameterTable")
    dataset.add_component("LanguageTable")
    dataset.write(FormTable=forms, ParameterTable=params, LanguageTable=langs)


if __name__ == "__main__":
    forms, params, langs = read_data("./ALT/raw/ALT-standardized_forms.csv")
    write_cldf(forms, params, langs)
