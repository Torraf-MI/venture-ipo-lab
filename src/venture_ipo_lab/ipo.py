"""IPO e private market: replica e sviluppo dell'analisi della tesi.

La tesi (USI, 2019) misurava con regressioni lineari semplici il legame tra l'attività di venture capital e
private equity e le IPO, su 20 e su 10 anni, in Europa occidentale e negli Stati Uniti, e concludeva che servivano
una regressione multivariata e i test F. Questo modulo:

1. replica le regressioni semplici sulle stesse finestre (1998-2018 e 2008-2018);
2. stima una regressione multivariata con i fattori che muovono il mercato delle IPO (valutazioni di borsa,
   volatilità, tassi) ed errori standard robusti (Newey-West);
3. verifica con un test F se il private market conta, a parità di condizioni di mercato, e se il suo effetto
   cambia dopo il 2008;
4. applica il test di Chow per una rottura strutturale nel 2008;
5. misura se le aziende arrivano in borsa sempre più tardi;
6. confronta underpricing e rendimento a 3 anni delle IPO con e senza VC o PE alle spalle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .dati import ANNO_ROTTURA

CONTROLLI = ["log_valutazione", "volatilita", "tasso_10a"]
NOMI = {"const": "Costante", "log_privato": "Capitale privato (log)", "log_valutazione": "Valutazioni di borsa (log)",
        "volatilita": "Volatilità", "tasso_10a": "Tasso a 10 anni",
        "log_privato_dopo": "Capitale privato (log) × dopo il 2008", "dopo": "Dopo il 2008",
        "operazioni_private": "Operazioni private", "capitale_privato": "Capitale privato"}


def prepara(serie: pd.DataFrame, regione: str, anno_rottura: int = ANNO_ROTTURA) -> pd.DataFrame:
    df = serie[serie["regione"] == regione].sort_values("anno").set_index("anno").copy()
    df["log_ipo"] = np.log(df["ipo"])
    df["log_privato"] = np.log(df["capitale_privato"])
    df["log_valutazione"] = np.log(df["valutazione"])
    df["dopo"] = (df.index >= anno_rottura).astype(float)
    df["log_privato_dopo"] = df["log_privato"] * df["dopo"]
    df["quota_backed"] = (df["ipo_vc"] + df["ipo_pe"]) / df["ipo"]
    return df


@dataclass
class Regressione:
    y: str
    x: list[str]
    inizio: int
    fine: int
    n: int
    coef: pd.Series
    t: pd.Series
    p: pd.Series
    r2: float
    r2_corretto: float
    f: float
    f_p: float
    risultato: object  # RegressionResults di statsmodels

    def tabella(self) -> pd.DataFrame:
        return pd.DataFrame({"Coefficiente": self.coef, "t": self.t, "p-value": self.p}).rename(index=NOMI)


def ols(df: pd.DataFrame, y: str, x: list[str], inizio: int | None = None, fine: int | None = None,
        hac: bool = True) -> Regressione:
    d = df.loc[(inizio or df.index.min()):(fine or df.index.max()), [y, *x]].dropna()
    X = sm.add_constant(d[x])
    fit = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2}) if hac else sm.OLS(d[y], X).fit()
    return Regressione(y, x, int(d.index.min()), int(d.index.max()), len(d), fit.params, fit.tvalues, fit.pvalues,
                       float(fit.rsquared), float(fit.rsquared_adj), float(np.asarray(fit.fvalue).squeeze()),
                       float(np.asarray(fit.f_pvalue).squeeze()), fit)


def replica_tesi(serie: pd.DataFrame, fine: int = 2018) -> pd.DataFrame:
    """Le regressioni semplici della tesi: IPO contro operazioni e capitale del private market, su 20 e 10 anni.

    Come nella tesi: errori standard classici, variabili in livelli.
    """
    righe = []
    for regione in serie["regione"].unique():
        df = prepara(serie, regione)
        for anni in (20, 10):
            for y, x, etichetta in (("ipo", "operazioni_private", "Numero di IPO ~ numero di operazioni private"),
                                    ("ipo_raccolta", "capitale_privato", "Raccolta IPO ~ capitale privato")):
                r = ols(df, y, [x], fine - anni, fine, hac=False)
                righe.append({"Regione": regione, "Finestra": f"{fine - anni}-{fine}", "Modello": etichetta,
                              "Osservazioni": r.n, "Pendenza": r.coef[x], "p-value": r.p[x], "R²": r.r2})
    return pd.DataFrame(righe)


def multivariata(df: pd.DataFrame, inizio: int = 1985, fine: int | None = None,
                 con_rottura: bool = True) -> Regressione:
    """log(IPO) su capitale privato e controlli di mercato; con ``con_rottura`` l'effetto del privato può
    cambiare dopo il 2008."""
    x = ["log_privato", *CONTROLLI]
    if con_rottura:
        x += ["dopo", "log_privato_dopo"]
    return ols(df, "log_ipo", x, inizio, fine)


def ftest_privato(df: pd.DataFrame, inizio: int = 1985, fine: int | None = None) -> pd.Series:
    """Test F: il private market (livello e cambio di effetto dopo il 2008) conta a parità di controlli?

    Confronta il modello completo con quello con i soli controlli di mercato (stime OLS classiche).
    """
    completo = multivariata(df, inizio, fine).risultato
    d = df.loc[inizio:(fine or df.index.max())]
    ristretto_x = [*CONTROLLI, "dopo"]
    ristretto = sm.OLS(d["log_ipo"], sm.add_constant(d[ristretto_x])).fit()
    pieno = sm.OLS(d["log_ipo"], sm.add_constant(d[["log_privato", *CONTROLLI, "dopo", "log_privato_dopo"]])).fit()
    f, p, q = pieno.compare_f_test(ristretto)
    # Solo il cambio di effetto dopo il 2008
    wald = pieno.f_test("log_privato_dopo = 0")
    return pd.Series({
        "F (privato conta)": f, "p-value (privato conta)": p, "Vincoli": q,
        "R² con privato": pieno.rsquared, "R² senza privato": ristretto.rsquared,
        "F (effetto cambia dopo il 2008)": float(np.asarray(wald.fvalue).squeeze()),
        "p-value (effetto cambia)": float(np.asarray(wald.pvalue).squeeze()),
        "Effetto prima del 2008": completo.params["log_privato"],
        "Effetto dopo il 2008": completo.params["log_privato"] + completo.params["log_privato_dopo"],
    })


def chow(df: pd.DataFrame, anno: int = ANNO_ROTTURA, inizio: int = 1985, fine: int | None = None,
              x: list[str] | None = None) -> pd.Series:
    """Test di Chow: i coefficienti del modello senza rottura sono gli stessi prima e dopo ``anno``?"""
    x = x or ["log_privato", *CONTROLLI]
    d = df.loc[inizio:(fine or df.index.max())]
    k = len(x) + 1

    def rss(parte: pd.DataFrame) -> float:
        return float(sm.OLS(parte["log_ipo"], sm.add_constant(parte[x])).fit().ssr)

    prima, dopo = d[d.index < anno], d[d.index >= anno]
    s_tot, s1, s2 = rss(d), rss(prima), rss(dopo)
    gl = len(d) - 2 * k
    f = ((s_tot - s1 - s2) / k) / ((s1 + s2) / gl)
    return pd.Series({"Anno di rottura": anno, "F di Chow": f, "p-value": 1 - stats.f.cdf(f, k, gl),
                      "Osservazioni prima": len(prima), "Osservazioni dopo": len(dopo)})


def ricerca_rottura(df: pd.DataFrame, anni: range = range(1995, 2016), inizio: int = 1985) -> pd.DataFrame:
    """F di Chow per ogni possibile anno di rottura: il massimo indica dove il legame cambia di più."""
    righe = [chow(df, a, inizio) for a in anni]
    return pd.DataFrame(righe).set_index("Anno di rottura")


def eta_alla_quotazione(serie: pd.DataFrame) -> pd.DataFrame:
    """Tendenza dell'età mediana alla IPO: anni in più per decennio e confronto prima/dopo il 2008."""
    righe = []
    for regione in serie["regione"].unique():
        df = prepara(serie, regione)
        r = ols(df.assign(t=df.index - df.index.min()), "eta_mediana_ipo", ["t"])
        righe.append({"Regione": regione, "Anni in più per decennio": 10 * r.coef["t"], "p-value": r.p["t"],
                      "Età media prima del 2008": df.loc[df.index < ANNO_ROTTURA, "eta_mediana_ipo"].mean(),
                      "Età media dal 2008": df.loc[df.index >= ANNO_ROTTURA, "eta_mediana_ipo"].mean()})
    return pd.DataFrame(righe).set_index("Regione")


def confronto_backing(ipo: pd.DataFrame) -> pd.DataFrame:
    """Underpricing e rendimento in eccesso a 3 anni (buy-and-hold contro il mercato) per regione e backing."""
    d = ipo.assign(bhar=ipo["rendimento_3a"] - ipo["mercato_3a"])
    g = d.groupby(["regione", "backing"])
    out = pd.DataFrame({
        "IPO": g.size(),
        "Underpricing medio": g["underpricing"].mean(),
        "Underpricing mediano": g["underpricing"].median(),
        "Rendimento in eccesso a 3 anni": g["bhar"].mean(),
        "Quota sotto il mercato": g["bhar"].apply(lambda s: (s < 0).mean()),
        "Età mediana": g["eta"].median(),
    })
    return out


def ttest_backing(ipo: pd.DataFrame, variabile: str = "underpricing") -> pd.DataFrame:
    """Differenza di media tra IPO con VC o PE e IPO senza backing: test t di Welch e test di Mann-Whitney."""
    d = ipo.assign(bhar=ipo["rendimento_3a"] - ipo["mercato_3a"])
    righe = []
    for regione in d["regione"].unique():
        base = d[(d["regione"] == regione) & (d["backing"] == "Nessuno")][variabile]
        for b in ("VC", "PE"):
            gruppo = d[(d["regione"] == regione) & (d["backing"] == b)][variabile]
            if len(gruppo) < 5 or len(base) < 5:
                continue
            t = stats.ttest_ind(gruppo, base, equal_var=False)
            mw = stats.mannwhitneyu(gruppo, base)
            righe.append({"Regione": regione, "Confronto": f"{b} contro nessun backing",
                          "Differenza": gruppo.mean() - base.mean(), "t di Welch": t.statistic,
                          "p-value t": t.pvalue, "p-value Mann-Whitney": mw.pvalue})
    return pd.DataFrame(righe)


def regressione_underpricing(ipo: pd.DataFrame) -> Regressione:
    """Underpricing su backing, età, dimensione e regione, con effetti fissi per anno (errori robusti)."""
    d = ipo.copy()
    d["VC"] = (d["backing"] == "VC").astype(float)
    d["PE"] = (d["backing"] == "PE").astype(float)
    d["log_eta"] = np.log(d["eta"])
    d["log_raccolta"] = np.log(d["raccolta"])
    d["Europa"] = (d["regione"] != "Stati Uniti").astype(float)
    anni = pd.get_dummies(d["anno"], prefix="a", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["VC", "PE", "log_eta", "log_raccolta", "Europa"]], anni], axis=1))
    fit = sm.OLS(d["underpricing"], X).fit(cov_type="HC1")
    keep = ["const", "VC", "PE", "log_eta", "log_raccolta", "Europa"]
    return Regressione("underpricing", keep[1:], int(d["anno"].min()), int(d["anno"].max()), len(d),
                       fit.params[keep], fit.tvalues[keep], fit.pvalues[keep], float(fit.rsquared),
                       float(fit.rsquared_adj), float(np.asarray(fit.fvalue).squeeze()),
                       float(np.asarray(fit.f_pvalue).squeeze()), fit)
