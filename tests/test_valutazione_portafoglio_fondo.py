import numpy as np
import pandas as pd
import pytest

from venture_ipo_lab import (
    DistribuzioneRendimenti,
    Scenario,
    concentrazione,
    first_chicago,
    flussi_fondo,
    indicatori,
    metodo_vc,
    piano_commitment,
    simula_portafogli,
)
from venture_ipo_lab.fondo import ParametriFondo, anno_pareggio, tir


def test_metodo_vc():
    e = metodo_vc(100_000_000, 2_000_000, multiplo_target=10, diluizioni_future=[0.2, 0.25])
    assert e.ritenzione == pytest.approx(0.6)
    assert e.post_money == pytest.approx(6_000_000)
    assert e.quota_richiesta == pytest.approx(1 / 3)
    assert metodo_vc(100, 1, tir_target=1.0, anni=3).multiplo_target == pytest.approx(8)
    with pytest.raises(ValueError):
        metodo_vc(10_000_000, 5_000_000, multiplo_target=10)


def test_first_chicago():
    sc = [Scenario("Zero", 0.5, 0, 3), Scenario("Bene", 0.5, 121, 2)]
    df, v = first_chicago(sc, 0.10)
    assert v == pytest.approx(50)
    with pytest.raises(ValueError):
        first_chicago([Scenario("x", 0.4, 1, 1)], 0.1)


def test_distribuzione_media_analitica_e_campione():
    d = DistribuzioneRendimenti()
    m = d.campiona(400_000, np.random.default_rng(1))
    assert m.min() >= 0 and m.max() <= d.tetto
    assert m.mean() == pytest.approx(d.media(), rel=0.03)


def test_diversificazione_riduce_la_perdita():
    p = simula_portafogli([5, 20, 100], simulazioni=4000)
    perdite = p["Prob. di perdere capitale"].to_numpy()
    assert perdite[0] > perdite[1] > perdite[2]
    assert (p["Multiplo mediano"] < p["Multiplo medio"]).all()  # asimmetria della legge di potenza


def test_concentrazione_ordinata():
    c = concentrazione(n=50_000)
    assert c.is_monotonic_increasing and c.iloc[-1] < 1


def test_takahashi_alexander():
    f = flussi_fondo(1.0)
    assert f["Richiami"].sum() == pytest.approx(1.0)
    assert f["NAV"].iloc[-1] == pytest.approx(0.0, abs=1e-12)
    ind = indicatori(f)
    assert ind["TIR"] == pytest.approx(ParametriFondo().crescita, abs=1e-6)  # proprietà del modello
    assert ind["TVPI"] > 1
    assert anno_pareggio(f) is not None and f["Flusso cumulato"].min() < 0


def test_pme_uguale_a_uno_se_il_fondo_replica_l_indice():
    p = ParametriFondo(crescita=0.08)
    f = flussi_fondo(1.0, p)
    indice = pd.Series(1.08 ** np.arange(1, p.vita + 1), index=f.index)
    assert indicatori(f, indice)["KS-PME"] == pytest.approx(1.0, rel=1e-9)


def test_tir_noto():
    assert tir([-100, 0, 121]) == pytest.approx(0.10)


def test_piano_commitment_raggiunge_obiettivo():
    piano, c0 = piano_commitment(1_000_000, 0.10, anni=20)
    assert c0 > 0
    assert piano["Quota NAV"].iloc[-1] == pytest.approx(0.10, rel=1e-6)
    assert (piano["Quota NAV + impegni"] >= piano["Quota NAV"]).all()
