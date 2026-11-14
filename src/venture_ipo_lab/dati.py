"""Dati del modulo sulle IPO: dataset dimostrativi sintetici e caricamento di dati propri da CSV.

I dataset dimostrativi **non sono dati reali**. Sono generati da un modello con un seme fisso, con ordini di
grandezza simili a quelli delle statistiche pubbliche sulle IPO e sul private market di Stati Uniti ed Europa
occidentale, e con relazioni note (che le analisi devono ritrovare). Servono a mostrare il metodo e a tenere i test
riproducibili. Per un'analisi vera si carica un CSV con le stesse colonne, ad esempio da un export Bloomberg o da
statistiche pubbliche.

Colonne della serie annuale (una riga per anno e regione):

``anno, regione, ipo, ipo_raccolta, ipo_vc, ipo_pe, operazioni_private, capitale_privato, eta_mediana_ipo,
tasso_10a, valutazione, volatilita``

Colonne del campione di IPO (una riga per IPO):

``anno, regione, backing (VC / PE / Nessuno), eta, raccolta, underpricing, rendimento_3a, mercato_3a``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGIONI = ["Stati Uniti", "Europa occidentale"]
COLONNE_SERIE = ["anno", "regione", "ipo", "ipo_raccolta", "ipo_vc", "ipo_pe", "operazioni_private",
                 "capitale_privato", "eta_mediana_ipo", "tasso_10a", "valutazione", "volatilita"]
COLONNE_IPO = ["anno", "regione", "backing", "eta", "raccolta", "underpricing", "rendimento_3a", "mercato_3a"]
ANNO_ROTTURA = 2008
ANNI = np.arange(1980, 2026)


def _interp(punti: dict[int, float]) -> np.ndarray:
    x = np.array(sorted(punti))
    return np.interp(ANNI, x, [punti[k] for k in x])


def genera_serie(seme: int = 2019) -> pd.DataFrame:
    """Serie annuali dimostrative 1980-2025 per Stati Uniti ed Europa occidentale (dati sintetici)."""
    rng = np.random.default_rng(seme)
    righe = []
    tassi = {"Stati Uniti": {1980: 11.5, 1984: 12.5, 1990: 8.5, 1998: 5.3, 2000: 6.0, 2003: 4.0, 2007: 4.6,
                             2012: 1.8, 2016: 1.8, 2019: 2.1, 2020: 0.9, 2021: 1.5, 2023: 4.0, 2025: 4.3},
             "Europa occidentale": {1980: 9.5, 1984: 10.5, 1990: 9.0, 1998: 4.6, 2000: 5.3, 2003: 4.1, 2007: 4.2,
                                    2012: 1.5, 2016: 0.4, 2019: 0.0, 2020: -0.3, 2021: 0.0, 2023: 2.6, 2025: 2.7}}
    valutazioni = {"Stati Uniti": {1980: 9, 1987: 15, 1990: 17, 1995: 22, 1999: 42, 2002: 24, 2007: 27, 2009: 15,
                                   2013: 23, 2017: 31, 2019: 30, 2021: 38, 2022: 28, 2025: 36},
                   "Europa occidentale": {1980: 8, 1987: 13, 1990: 14, 1995: 16, 1999: 30, 2002: 16, 2007: 20,
                                          2009: 11, 2013: 16, 2017: 19, 2019: 18, 2021: 22, 2022: 16, 2025: 20}}
    shock_vol = {1987: 14, 1990: 5, 1998: 8, 2001: 9, 2002: 12, 2008: 22, 2009: 12, 2011: 8, 2020: 13, 2022: 8}
    scala = {"Stati Uniti": 1.0, "Europa occidentale": 0.75}
    inizio_privato = {"Stati Uniti": 6.0, "Europa occidentale": 1.2}  # miliardi nel 1980

    for regione in REGIONI:
        tasso = _interp(tassi[regione]) + rng.normal(0, 0.25, len(ANNI))
        val = _interp(valutazioni[regione]) * np.exp(rng.normal(0, 0.05, len(ANNI)))
        vol = 17 + np.array([shock_vol.get(a, 0) for a in ANNI]) + rng.normal(0, 1.5, len(ANNI))
        # Capitale privato (VC + PE): crescita esponenziale, cicli e picco del 2021.
        crescita = np.where(ANNI < 2000, 0.13, 0.10) + rng.normal(0, 0.05, len(ANNI))
        crescita[ANNI == 2001] -= 0.45
        crescita[ANNI == 2009] -= 0.35
        crescita[ANNI == 2021] += 0.45
        crescita[ANNI == 2022] -= 0.30
        crescita[ANNI == 2023] -= 0.20
        if regione == "Europa occidentale":
            crescita = crescita + np.where(ANNI < 1998, 0.03, 0.0)
        privato = inizio_privato[regione] * np.exp(np.cumsum(crescita))
        operazioni = 120 * scala[regione] * (privato / privato[0]) ** 0.75 * np.exp(rng.normal(0, 0.08, len(ANNI)))

        # Numero di IPO: valutazioni (+), volatilità (−), tassi (−) e mercato privato.
        # Prima del 2008 IPO e capitale privato crescono insieme; dopo, il capitale privato sostituisce la borsa.
        lp = np.log(privato / privato[0])
        dopo = ANNI >= ANNO_ROTTURA
        effetto_privato = np.where(dopo, 0.18 * lp[ANNI == ANNO_ROTTURA][0] - 0.55 * (lp - lp[dopo][0]), 0.18 * lp)
        log_ipo = (np.log(85 * scala[regione]) + 1.15 * np.log(val / 10) - 0.035 * (vol - 17) - 0.06 * tasso
                   + effetto_privato + rng.normal(0, 0.18, len(ANNI)))
        ipo = np.maximum(np.round(np.exp(log_ipo)), 8)
        dim_media = 40 * np.exp(0.055 * (ANNI - 1980)) * (val / 20) ** 0.5  # milioni
        quota_vc = np.clip(0.22 + 0.006 * (ANNI - 1980) + rng.normal(0, 0.03, len(ANNI)), 0.05, 0.6)
        quota_pe = np.clip(0.04 + 0.007 * (ANNI - 1980) + rng.normal(0, 0.03, len(ANNI)), 0.01, 0.5)
        if regione == "Europa occidentale":
            quota_vc, quota_pe = quota_vc * 0.6, quota_pe * 1.2
        eta = (6.5 if regione == "Stati Uniti" else 8.5) + 0.09 * (ANNI - 1980) + 2.5 * dopo \
            + rng.normal(0, 0.6, len(ANNI))
        for i, anno in enumerate(ANNI):
            righe.append({
                "anno": int(anno), "regione": regione, "ipo": int(ipo[i]),
                "ipo_raccolta": float(ipo[i] * dim_media[i] / 1000),  # miliardi
                "ipo_vc": int(round(ipo[i] * quota_vc[i])), "ipo_pe": int(round(ipo[i] * quota_pe[i])),
                "operazioni_private": int(round(operazioni[i])), "capitale_privato": float(privato[i]),
                "eta_mediana_ipo": float(eta[i]), "tasso_10a": float(tasso[i]), "valutazione": float(val[i]),
                "volatilita": float(vol[i]),
            })
    return pd.DataFrame(righe, columns=COLONNE_SERIE)


def genera_ipo(serie: pd.DataFrame | None = None, dal: int = 2000, frazione: float = 0.35,
               seme: int = 1912) -> pd.DataFrame:
    """Campione dimostrativo di singole IPO con backing, underpricing e rendimento a 3 anni (dati sintetici)."""
    serie = genera_serie() if serie is None else serie
    rng = np.random.default_rng(seme)
    parametri = {  # (underpricing medio, rendimento in eccesso a 3 anni medio)
        ("Stati Uniti", "VC"): (0.24, -0.06), ("Stati Uniti", "PE"): (0.11, 0.03),
        ("Stati Uniti", "Nessuno"): (0.15, -0.17),
        ("Europa occidentale", "VC"): (0.13, -0.08), ("Europa occidentale", "PE"): (0.06, 0.01),
        ("Europa occidentale", "Nessuno"): (0.09, -0.13),
    }
    righe = []
    for _, r in serie[(serie["anno"] >= dal) & (serie["anno"] <= 2022)].iterrows():
        n = rng.binomial(int(r["ipo"]), frazione)
        p_vc, p_pe = r["ipo_vc"] / r["ipo"], r["ipo_pe"] / r["ipo"]
        mercato = rng.normal(0.25, 0.25)  # rendimento del mercato a 3 anni, comune per anno
        for _ in range(n):
            u = rng.random()
            backing = "VC" if u < p_vc else ("PE" if u < p_vc + p_pe else "Nessuno")
            up_medio, eccesso = parametri[(r["regione"], backing)]
            caldo = 0.10 * (r["valutazione"] / serie["valutazione"].median() - 1)  # mercati caldi: più underpricing
            underpricing = up_medio + caldo + rng.standard_t(4) * 0.12
            eta = max(1.0, r["eta_mediana_ipo"] * np.exp(rng.normal(0, 0.45))
                      * (0.8 if backing == "VC" else 1.3 if backing == "PE" else 1.0))
            raccolta = r["ipo_raccolta"] / r["ipo"] * 1000 * np.exp(rng.normal(0, 0.8)) \
                * (1.6 if backing == "PE" else 1.0)
            rend = (1 + mercato) * np.exp(eccesso + rng.normal(0, 0.55) - 0.55 ** 2 / 2) - 1
            righe.append({"anno": int(r["anno"]), "regione": r["regione"], "backing": backing, "eta": eta,
                          "raccolta": raccolta, "underpricing": underpricing, "rendimento_3a": rend,
                          "mercato_3a": mercato})
    return pd.DataFrame(righe, columns=COLONNE_IPO)


def carica_serie(percorso: str | Path) -> pd.DataFrame:
    df = pd.read_csv(percorso)
    mancanti = set(COLONNE_SERIE) - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti nel CSV della serie: {', '.join(sorted(mancanti))}")
    return df[COLONNE_SERIE].sort_values(["regione", "anno"]).reset_index(drop=True)


def carica_ipo(percorso: str | Path) -> pd.DataFrame:
    df = pd.read_csv(percorso)
    mancanti = set(COLONNE_IPO) - set(df.columns)
    if mancanti:
        raise ValueError(f"Colonne mancanti nel CSV delle IPO: {', '.join(sorted(mancanti))}")
    return df[COLONNE_IPO]
