import re
from pycldf import Wordlist, StructureDataset


def tokenize(word):
    tokens = re.findall(".ː?ʲ?", word)
    assert "".join(tokens) == word

    return tokens


"""
read the data from a csv file, return forms, languages and parameters as required by the CLDF standard
"""
def read_data(fp):
    forms, params, langs = [], [], []
    word_id = 10000
    with open("ALT/raw/ALT-standardized_forms.csv") as f:
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
    dataset = Wordlist.in_dir(output_dir)
    dataset.add_component("ParameterTable")
    dataset.add_component("LanguageTable")
    dataset.write(FormTable=forms, ParameterTable=params, LanguageTable=langs)


if __name__ == "__main__":
    forms, params, langs = read_data("./ALT/raw/ALT-standardized_forms.csv")
    write_cldf(forms, params, langs)
