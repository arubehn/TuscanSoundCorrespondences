from shibboleth import ShibbolethCalculator
import pytest


@pytest.fixture
def calculator(test_data):
    varieties = ["101", "102", "103"]
    data_fp = test_data / "test_cldf/Wordlist-metadata.json"

    sc = ShibbolethCalculator(varieties, data_fp)

    return sc

def test_align(calculator):
    assert calculator.data[10005][13] == ['a', 's', 'd', 'o', '-']
    assert calculator.data[10005][6] == ['a', 's', 'd', 'o']

def test_metrics(calculator):
    charac, repr, dist = calculator.calculate_metrics(normalize=True)

    # check relevant representativeness values
    assert repr["[t] : [d]"] == 1.0
    assert repr["[r] : [s]"] == pytest.approx(0.857, 0.001)
    assert (repr["[e] : [i]"] == repr["[e] : [o]"] == repr["[r] : [l]"] == repr["[v] : [-]"]
            == pytest.approx(0.744, 0.001))

    # check relevant distinctiveness values
    assert dist["[t] : [d]"] == 1.0
    assert dist["[r] : [s]"] == pytest.approx(0.799, 0.001)
    assert (dist["[e] : [i]"] == dist["[e] : [o]"] == dist["[r] : [l]"] == dist["[v] : [-]"]
            == pytest.approx(0.594, 0.001))

    # check relevant characteristicness values
    assert charac["[t] : [d]"] == 1.0
    assert charac["[r] : [s]"] == pytest.approx(0.827, 0.001)
    assert (charac["[e] : [i]"] == charac["[e] : [o]"] == charac["[r] : [l]"] == charac["[v] : [-]"]
            == pytest.approx(0.661, 0.001))
