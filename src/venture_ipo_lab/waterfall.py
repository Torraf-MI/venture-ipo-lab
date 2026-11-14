"""Distribuzione dei proventi all'exit (waterfall) con liquidation preference.

Regole:

* le classi privilegiate sono pagate per seniority (prima la più alta); a parità di seniority *pari passu*, in
  proporzione alla preference;
* una privilegiata **non partecipante** sceglie il maggiore tra la preference e la conversione in ordinarie;
* una **partecipante** prende la preference e poi partecipa al residuo come le ordinarie; se ha un tetto
  (``cap_partecipazione``), riceve al massimo ``cap × investito`` oppure converte, se convertire rende di più;
* le scelte di conversione sono un equilibrio: nessuna classe può guadagnare cambiando la propria scelta;
* l'option pool è trattato come interamente assegnato ed esercitato a costo zero (ipotesi prudente per i
  fondatori: nella realtà le opzioni non assegnate non partecipano).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cap_table import COMUNI, CapTable


def _pagamenti(ct: CapTable, valore: float, convertite: set[str]) -> dict[str, float]:
    """Pagamento per classe data una scelta di conversione."""
    restante = valore
    paga = {c: 0.0 for c in ct.classi}
    pref = {c: k.multiplo_pref * k.investito for c, k in ct.classi.items()
            if k.privilegiata and c not in convertite}
    for liv in sorted({ct.classi[c].seniority for c in pref}, reverse=True):
        gruppo = [c for c in pref if ct.classi[c].seniority == liv]
        dovuto = sum(pref[c] for c in gruppo)
        pagato = min(restante, dovuto)
        for c in gruppo:
            paga[c] += pagato * pref[c] / dovuto if dovuto > 0 else 0.0
        restante -= pagato

    # Residuo alle ordinarie, alle convertite e alle partecipanti (con eventuale tetto).
    partecipano = {c: ct.azioni_classe(c) for c, k in ct.classi.items()
                   if c == COMUNI or not k.privilegiata or c in convertite or k.partecipante}
    tetto = {c: ct.classi[c].cap_partecipazione * ct.classi[c].investito - paga[c] for c in partecipano
             if c not in convertite and ct.classi[c].partecipante and ct.classi[c].cap_partecipazione}
    attive = dict(partecipano)
    while restante > 1e-9 and attive:
        tot = sum(attive.values())
        quota = {c: restante * a / tot for c, a in attive.items()}
        sforano = [c for c in attive if c in tetto and quota[c] > tetto[c] + 1e-9]
        if not sforano:
            for c, q in quota.items():
                paga[c] += q
            restante = 0.0
            break
        for c in sforano:
            paga[c] += max(tetto[c], 0.0)
            restante -= max(tetto[c], 0.0)
            del attive[c]
    return paga


def _convertibili(ct: CapTable) -> list[str]:
    return [c for c, k in ct.classi.items()
            if k.privilegiata and (not k.partecipante or k.cap_partecipazione is not None)]


def distribuisci_classi(ct: CapTable, valore: float) -> pd.Series:
    """Proventi per classe a un dato valore di exit (al netto del debito), con conversioni in equilibrio."""
    conv: set[str] = set()
    candidati = _convertibili(ct)
    # Risposta ottima iterata: in ordine di prezzo crescente, ogni classe converte se le conviene.
    for _ in range(4 * len(candidati) + 4):
        cambiato = False
        for c in sorted(candidati, key=lambda c: ct.classi[c].prezzo):
            base = _pagamenti(ct, valore, conv)[c]
            alt = _pagamenti(ct, valore, conv ^ {c})[c]
            if alt > base + 1e-6:
                conv ^= {c}
                cambiato = True
        if not cambiato:
            break
    paga = _pagamenti(ct, valore, conv)
    s = pd.Series(paga, name=valore)
    s.attrs["convertite"] = sorted(conv)
    return s


def distribuisci(ct: CapTable, valore: float) -> pd.Series:
    """Proventi per titolare a un dato valore di exit."""
    per_classe = distribuisci_classi(ct, valore)
    out: dict[str, float] = {}
    for (t, c), a in ct.posizioni.items():
        tot = ct.azioni_classe(c)
        out[t] = out.get(t, 0.0) + (per_classe[c] * a / tot if tot > 0 else 0.0)
    return pd.Series(out, name=valore)


def tabella_exit(ct: CapTable, valori: list[float] | np.ndarray, per: str = "titolare") -> pd.DataFrame:
    """Proventi per titolare (o per classe) su una griglia di valori di exit (righe = valori)."""
    f = distribuisci if per == "titolare" else distribuisci_classi
    return pd.DataFrame([f(ct, v) for v in valori], index=pd.Index(valori, name="Valore di exit"))


def multipli_exit(ct: CapTable, valore: float) -> pd.DataFrame:
    """Per ogni titolare: investito, proventi, multiplo e quota dei proventi contro quota azionaria."""
    tab = ct.tabella()
    inv = tab.groupby("Titolare")["Investito"].sum()
    quota = tab.groupby("Titolare")["Quota"].sum()
    pro = distribuisci(ct, valore)
    df = pd.DataFrame({"Investito": inv, "Quota azionaria": quota, "Proventi": pro})
    df["Quota dei proventi"] = df["Proventi"] / valore if valore > 0 else 0.0
    df["Multiplo"] = np.where(df["Investito"] > 0, df["Proventi"] / df["Investito"].where(df["Investito"] > 0), np.nan)
    return df.sort_values("Proventi", ascending=False)


def soglia_conversione(ct: CapTable, classe: str, massimo: float | None = None) -> float | None:
    """Il valore di exit minimo oltre il quale ``classe`` preferisce convertire in ordinarie (bisezione)."""
    if classe not in _convertibili(ct):
        return None
    hi = massimo or ct.round_info[-1]["Post-money"] * 50
    if classe not in distribuisci_classi(ct, hi).attrs["convertite"]:
        return None
    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if classe in distribuisci_classi(ct, mid).attrs["convertite"]:
            hi = mid
        else:
            lo = mid
    return hi
