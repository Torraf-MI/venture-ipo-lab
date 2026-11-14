import numpy as np
import pytest

from venture_ipo_lab import CASO_DEMO, CapTable, costruisci_cap_table, distribuisci, distribuisci_classi
from venture_ipo_lab.cap_table import POOL
from venture_ipo_lab.waterfall import multipli_exit, soglia_conversione


def semplice(pref=1.0, partecipante=False, cap=None):
    ct = CapTable.fondazione({"F": 1.0}, azioni=1_000_000)
    ct.round("A", 4_000_000, {"VC": 1_000_000}, multiplo_pref=pref, partecipante=partecipante,
             cap_partecipazione=cap)
    return ct


def test_round_prezzo_e_quote():
    ct = semplice()
    assert ct.rounds().loc["A", "Prezzo per azione"] == pytest.approx(4.0)
    q = ct.quote_per_titolare()
    assert q["VC"] == pytest.approx(0.2)
    assert q.sum() == pytest.approx(1.0)


def test_option_pool_shuffle_raggiunge_target_e_diluisce_solo_i_soci():
    ct = CapTable.fondazione({"F": 1.0}, azioni=1_000_000)
    ct.round("A", 4_000_000, {"VC": 1_000_000}, pool_post=0.15)
    q = ct.quote_per_titolare()
    assert q[POOL] == pytest.approx(0.15)
    assert q["VC"] == pytest.approx(0.2)  # l'investitore ha esattamente la quota pagata
    assert ct.rounds().loc["A", "Pre-money effettivo"] == pytest.approx(4_000_000 - 0.15 * 5_000_000)


def test_safe_post_money_al_cap():
    ct = CapTable.fondazione({"F": 1.0}, azioni=1_000_000)
    ct.safe("Angel", 300_000, cap=3_000_000)
    ct.round("Seed", 6_000_000, {"VC": 1_000_000})
    q = ct.quote_per_titolare()
    # Al cap, 10% del capitale prima del round, poi diluito dal round (post = 7 mln, round = 1/7)
    assert q["Angel"] == pytest.approx(0.10 * 6 / 7, rel=1e-6)
    assert q["VC"] == pytest.approx(1 / 7, rel=1e-6)


def test_safe_con_sconto_quando_il_cap_e_alto():
    ct = CapTable.fondazione({"F": 1.0}, azioni=1_000_000)
    ct.safe("Angel", 200_000, cap=50_000_000, sconto=0.2)
    ct.round("Seed", 4_000_000, {"VC": 1_000_000})
    prezzo = ct.rounds().loc["Seed", "Prezzo per azione"]
    azioni = ct.posizioni[("Angel", "Seed")]
    assert 200_000 / azioni == pytest.approx(prezzo * 0.8, rel=1e-6)


def test_safe_senza_cap_ne_sconto_rifiutato():
    ct = CapTable.fondazione({"F": 1.0})
    with pytest.raises(ValueError):
        ct.safe("Angel", 100_000)


@pytest.mark.parametrize("valore", [0, 500_000, 1_000_000, 3_000_000, 5_000_000, 20_000_000])
def test_waterfall_distribuisce_tutto(valore):
    ct = costruisci_cap_table(CASO_DEMO)
    assert distribuisci(ct, valore).sum() == pytest.approx(valore, abs=1e-3)


def test_non_partecipante_prende_preference_o_converte():
    ct = semplice()
    assert distribuisci(ct, 800_000)["VC"] == pytest.approx(800_000)  # sotto la preference prende tutto
    assert distribuisci(ct, 3_000_000)["VC"] == pytest.approx(1_000_000)  # preference
    assert distribuisci(ct, 20_000_000)["VC"] == pytest.approx(4_000_000)  # converte: 20%
    assert soglia_conversione(ct, "A") == pytest.approx(5_000_000, rel=1e-6)  # pari al post-money


def test_partecipante_doppio_dip_e_tetto():
    ct = semplice(partecipante=True)
    assert distribuisci(ct, 3_000_000)["VC"] == pytest.approx(1_000_000 + 0.2 * 2_000_000)
    capped = semplice(partecipante=True, cap=2.0)
    assert distribuisci(capped, 8_000_000)["VC"] == pytest.approx(2_000_000)  # tetto 2x
    assert distribuisci(capped, 50_000_000)["VC"] == pytest.approx(10_000_000)  # converte sopra il tetto


def test_seniority_paga_prima_il_round_piu_recente():
    ct = CapTable.fondazione({"F": 1.0}, azioni=1_000_000)
    ct.round("A", 4_000_000, {"A": 1_000_000})
    ct.round("B", 10_000_000, {"B": 2_000_000})
    p = distribuisci_classi(ct, 2_500_000)
    assert p["B"] == pytest.approx(2_000_000)
    assert p["A"] == pytest.approx(500_000)


def test_multipli_usano_il_versato_effettivo():
    ct = costruisci_cap_table(CASO_DEMO)
    m = multipli_exit(ct, 220_000_000)
    assert m.loc["Business angel", "Investito"] == pytest.approx(250_000)
    assert m.loc["Fondo seed", "Investito"] == pytest.approx(1_700_000)
    assert np.isnan(m.loc["Fondatrice A", "Multiplo"])
