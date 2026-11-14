import pandas as pd
import pytest

from venture_ipo_lab import empirico


@pytest.fixture(scope="module")
def analisi():
    return empirico.analizza()


def test_dati_pubblici_coerenti():
    d = empirico.carica_dati()
    assert d.index.min() == 1980 and d.index.max() == 2025
    assert d.loc[1999, "ipo"] == 476 and d.loc[2008, "ipo"] == 21  # valori della tabella 1 delle statistiche IPO
    c = d.loc[1980:2025]
    assert (c["ipo_vc"] <= c["ipo_sponsor_finanziario"]).all()
    assert (c["ipo_sponsor_finanziario"] <= c["ipo"]).all()
    assert d["vix"].loc[:1989].isna().all() and d["vix"].loc[1990:].notna().all()


def test_risultati_principali(analisi):
    assert analisi.tabella_tesi()["R²"].max() < 0.3
    assert analisi.f_controlli["p-value"] < 0.001
    assert analisi.f_mercati_privati["p-value"] > 0.1
    assert analisi.chow_2008["p-value"] < 0.01
    assert analisi.rottura["sup-F"] > analisi.rottura["critico 1%"]
    assert analisi.eta["media dopo"] > analisi.eta["media prima"]


def test_chow_e_sup_f_su_serie_stabile():
    import numpy as np
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=60)}, index=range(1960, 2020))
    df["y"] = 1 + 2 * df["x"] + rng.normal(0, 0.5, 60)
    assert empirico.chow(df, "y", ["x"], 1990)["p-value"] > 0.05
    r = empirico.sup_f(df, "y", ["x"])
    assert r["sup-F"] < r["critico 1%"]


def test_carica_export(tmp_path):
    p = tmp_path / "e.csv"
    pd.DataFrame({"Year": [2000, 2001], "IPO": [10, 12]}).to_csv(p, index=False)
    e = empirico.carica_export(str(p), {"Year": "anno", "IPO": "ipo"})
    assert list(e.index) == [2000, 2001] and e.loc[2001, "ipo"] == 12
