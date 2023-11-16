import os
from shibboleth.calculate import *
from os import path


def test_read_areas_from_file(test_data):
    ref = {
        100: 1,
        94: 2,
        99: 1,
        42: 3
    }

    assert read_areas_from_file(test_data / "test_areas.tsv") == ref


def test_load_wordlist(test_data):
    data = load_data(test_data / "test_wordlist.tsv")

    assert data.rows == ["bischero"]
    assert data.cols[0] == "101"

    assert data.height == 1
    assert data.width == 6
    assert len(data) == 6


def test_load_wordlist_from_metadata(test_data):
    data = load_data(test_data / "test_cldf/Wordlist-metadata.json")

    assert data.rows == ["bischero"]
    assert data.cols[0] == "101"

    assert data.height == 1
    assert data.width == 6
    assert len(data) == 6


def test_write_results_to_file(test_data):
    reference_file = test_data / "test_results.txt"
    result_file = test_data / "test_results_TMP.txt"

    # store current (reference) contents of results file
    with open(reference_file) as f:
        ref = f.read()

    # re-calculate the results
    varieties = ["101", "102", "104"]
    data_fp = test_data / "test_cldf/Wordlist-metadata.json"

    sc = ShibbolethCalculator(varieties, data_fp)
    charac, repr, dist = sc.calculate_metrics(normalize=True)
    freq = sc.get_frequencies()

    # write results to file
    write_results_to_file(result_file, charac, repr, dist, freq)

    # check whether file was created properly
    assert path.isfile(result_file)

    # read in contents and check whether they are identical with the reference file
    with open(result_file) as f:
        res = f.read()

    assert res == ref

    # delete temporary result file
    os.remove(result_file)
