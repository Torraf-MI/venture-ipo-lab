import json

import pandas as pd
import pytest

from venture_ipo_lab import (
    CASO_DEMO,
    chow,
    confronto_backing,
    eta_alla_quotazione,
    ftest_privato,
    genera_ipo,
    genera_serie,
    prepara,
    replica_tesi,
    ttest_backing,
)
from venture_ipo_lab.cli import main
from venture_ipo_lab.dati import COLONNE_SERIE, carica_serie


@pytest.fixture(scope="module")
def serie():
    return genera_serie()


def test_serie_riproducibile_e_completa(serie):
    assert list(serie.columns) == COLONNE_SERIE
    assert len(serie) == 2 * 46
    pd.testing.assert_frame_equal(serie, genera_serie())
    assert (serie["ipo_vc"] + serie["ipo_pe"] <= serie["ipo"]).all()


def test_replica_tesi_r2_basso_su_20_anni(serie):
    r = replica_tesi(serie)
    conteggi = r[r["Modello"].str.startswith("Numero")]
    venti = conteggi[conteggi["Finestra"] == "1998-2018"]["R²"]
    dieci = conteggi[conteggi["Finestra"] == "2008-2018"]["R²"]
    assert venti.max() < 0.1 and (dieci.to_numpy() > venti.to_numpy()).all()


@pytest.mark.parametrize("regione", ["Stati Uniti", "Europa occidentale"])
def test_multivariata_ritrova_il_cambio_di_segno(serie, regione):
    t = ftest_privato(prepara(serie, regione))
    assert t["p-value (privato conta)"] < 0.01
    assert t["Effetto prima del 2008"] > 0 > t["Effetto dopo il 2008"]
    assert chow(prepara(serie, regione))["p-value"] < 0.05


def test_chow_senza_rottura_non_rifiuta():
    import numpy as np
    rng = np.random.default_rng(0)
    anni = np.arange(1985, 2026)
    df = pd.DataFrame({"log_privato": rng.normal(size=41), "log_valutazione": rng.normal(size=41),
                       "volatilita": rng.normal(size=41), "tasso_10a": rng.normal(size=41)}, index=anni)
    df["log_ipo"] = 1 + 0.5 * df["log_privato"] - df["volatilita"] + rng.normal(0, 0.3, 41)
    assert chow(df)["p-value"] > 0.05


def test_eta_crescente(serie):
    e = eta_alla_quotazione(serie)
    assert (e["Anni in più per decennio"] > 0).all()
    assert (e["Età media dal 2008"] > e["Età media prima del 2008"]).all()


def test_campione_ipo_e_confronti(serie):
    ipo = genera_ipo(serie)
    c = confronto_backing(ipo)
    assert c.loc[("Stati Uniti", "VC"), "Underpricing medio"] > c.loc[("Stati Uniti", "Nessuno"), "Underpricing medio"]
    t = ttest_backing(ipo)
    assert set(t["Confronto"]) == {"VC contro nessun backing", "PE contro nessun backing"}


def test_carica_serie_csv(tmp_path, serie):
    p = tmp_path / "s.csv"
    serie.to_csv(p, index=False)
    pd.testing.assert_frame_equal(carica_serie(p), serie.sort_values(["regione", "anno"]).reset_index(drop=True))
    serie.drop(columns="ipo").to_csv(p, index=False)
    with pytest.raises(ValueError):
        carica_serie(p)


def test_cli_demo_e_caso_json(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["--demo", "--simulazioni", "1000", "--output", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "Dalla start-up alla quotazione" in html and "Dati dimostrativi" in html
    caso = tmp_path / "c.json"
    caso.write_text(json.dumps(CASO_DEMO), encoding="utf-8")
    assert main(["--caso", str(caso), "--simulazioni", "1000", "--output", str(tmp_path / "c.html")]) == 0
    assert "Report:" in capsys.readouterr().out
