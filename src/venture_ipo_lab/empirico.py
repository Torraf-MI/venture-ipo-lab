"""Modulo empirico: IPO e mercati privati, 1980-2025, con dati pubblici.

Riprende l'analisi della tesi (numero di IPO e attività di private equity e venture capital, Stati Uniti ed Europa
occidentale) e svolge i passi che la tesi indicava come sviluppo:

* le regressioni lineari semplici della tesi, rifatte su dati pubblici e sugli ultimi undici anni (2015-2025);
* una regressione multivariata con i controlli di mercato (tassi, valutazioni di borsa, volatilità, rendimento
  azionario) e i test F di significatività congiunta;
* la rottura strutturale intorno al 2008 (test di Chow) e la ricerca della data di rottura (sup-F di Quandt e
  Andrews);
* la verifica dell'ipotesi "le aziende restano private più a lungo" sull'età alla quotazione;
* l'underpricing del primo giorno e il legame con la quota di IPO sostenute da VC e PE.

Le regressioni sulle serie annuali usano errori standard robusti all'autocorrelazione (Newey-West).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------------------------- dati


def _csv(nome: str) -> pd.DataFrame:
    with resources.files("venture_ipo_lab.dati_pubblici").joinpath(nome).open("r", encoding="utf-8") as f:
        return pd.read_csv(f, index_col="anno")


def carica_dati() -> pd.DataFrame:
    """Tutte le serie annuali in un'unica tabella indicizzata per anno, con le variabili derivate."""
    df = _csv("ipo_usa.csv").join(_csv("macro.csv"), how="outer").join(_csv("mercati_privati.csv"), how="outer")
    df["quota_vc"] = df["ipo_vc"] / df["ipo"]
    df["quota_sponsor"] = df["ipo_sponsor_finanziario"] / df["ipo"]
    df["ipo_non_sponsor"] = df["ipo"] - df["ipo_sponsor_finanziario"]
    df["ln_ipo"] = np.log(df["ipo"])
    df["ln_cape"] = np.log(df["cape_gennaio"])
    df["sp500_anno_prima"] = df["sp500_rendimento_totale"].shift(1)
    df["dax_anno_prima"] = df["dax_variazione"].shift(1)
    df["dopo_2008"] = (df.index >= 2008).astype(float)
    df["bolla_dotcom"] = df.index.isin([1999, 2000]).astype(float)
    df["eur_quota_ipo_exit_pe"] = df["eur_pe_ipo_numero"] / df["eur_pe_exit_numero"]
    df["usa_ipo_vc_per_mille_round"] = 1000 * df["ipo_vc"] / df["usa_vc_operazioni"]
    return df


COLONNE_EXPORT = {"anno": "anno", "ipo": "ipo", "ipo_vc": "ipo_vc", "ipo_pe": "ipo_buyout",
                  "operazioni_pe": "operazioni_pe", "operazioni_vc": "operazioni_vc",
                  "valore_pe": "valore_pe", "valore_vc": "valore_vc"}


def carica_export(percorso: str, mappa: dict[str, str] | None = None, foglio: str | int = 0) -> pd.DataFrame:
    """Carica un export annuale (per esempio da Bloomberg) con proprie serie di IPO e operazioni di PE/VC.

    ``mappa`` associa i nomi delle colonne del file ai nomi standard (``anno``, ``ipo``, ``ipo_vc``, ``ipo_pe``,
    ``operazioni_pe``, ``operazioni_vc``, ``valore_pe``, ``valore_vc``); colonne in più restano disponibili.
    Il risultato si può unire a :func:`carica_dati` o usare direttamente nelle funzioni di regressione.
    """
    if str(percorso).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(percorso, sheet_name=foglio)
    else:
        df = pd.read_csv(percorso)
    if mappa:
        df = df.rename(columns=mappa)
    if "anno" not in df.columns:
        raise ValueError("Il file deve avere una colonna con l'anno (o una mappa verso 'anno')")
    return df.set_index("anno").sort_index()


# ---------------------------------------------------------------------------- regressioni


@dataclass
class Regressione:
    nome: str
    y: str
    x: list[str]
    risultato: object
    campione: tuple[int, int]

    @property
    def n(self) -> int:
        return int(self.risultato.nobs)

    @property
    def r2(self) -> float:
        return float(self.risultato.rsquared)

    def tabella(self) -> pd.DataFrame:
        r = self.risultato
        return pd.DataFrame({"Coefficiente": r.params, "Errore std": r.bse, "t": r.tvalues, "p-value": r.pvalues})

    def test_f(self, variabili: list[str]) -> dict:
        """Test F (robusto) che i coefficienti delle variabili siano tutti nulli."""
        r = self.risultato
        vincoli = np.zeros((len(variabili), len(r.params)))
        for i, v in enumerate(variabili):
            vincoli[i, list(r.params.index).index(v)] = 1.0
        t = r.f_test(vincoli)
        return {"F": float(np.squeeze(t.fvalue)), "p-value": float(t.pvalue), "gdl": (int(t.df_num), int(t.df_denom))}


def ols(df: pd.DataFrame, y: str, x: list[str], dal: int | None = None, al: int | None = None,
        nome: str = "", hac: bool = True, lag: int | None = None) -> Regressione:
    d = df.loc[dal:al, [y, *x]].dropna()
    X = sm.add_constant(d[x], has_constant="add")
    m = sm.OLS(d[y], X)
    if hac:
        lag = lag if lag is not None else max(1, int(np.floor(4 * (len(d) / 100) ** (2 / 9))))
        r = m.fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
    else:
        r = m.fit()
    return Regressione(nome or f"{y} ~ {' + '.join(x)}", y, x, r, (int(d.index.min()), int(d.index.max())))


def chow(df: pd.DataFrame, y: str, x: list[str], anno: int, dal: int | None = None,
         al: int | None = None) -> dict:
    """Test di Chow di rottura strutturale (tutti i coefficienti) a partire da ``anno``."""
    d = df.loc[dal:al, [y, *x]].dropna()
    k = len(x) + 1

    def ssr(parte):
        return sm.OLS(parte[y], sm.add_constant(parte[x], has_constant="add")).fit().ssr

    a, b = d[d.index < anno], d[d.index >= anno]
    if len(a) <= k or len(b) <= k:
        raise ValueError("Troppe poche osservazioni in uno dei due sottoperiodi")
    s_tot, s_a, s_b = ssr(d), ssr(a), ssr(b)
    gdl = len(d) - 2 * k
    f = ((s_tot - s_a - s_b) / k) / ((s_a + s_b) / gdl)
    return {"anno": anno, "F": float(f), "p-value": float(stats.f.sf(f, k, gdl)), "gdl": (k, gdl),
            "n prima": len(a), "n dopo": len(b)}


# valori critici asintotici della statistica QLR / sup-F in forma F (trimming 15%), per numero di coefficienti
# sottoposti a test: Andrews (1993), nella versione tabulata da Stock e Watson (10%, 5%, 1%)
_ANDREWS_15 = {1: (7.12, 8.68, 12.16), 2: (5.00, 5.86, 7.78), 3: (4.09, 4.71, 6.02), 4: (3.59, 4.09, 5.12),
               5: (3.26, 3.66, 4.53), 6: (3.02, 3.37, 4.12), 7: (2.84, 3.15, 3.82), 8: (2.69, 2.98, 3.57)}


def sup_f(df: pd.DataFrame, y: str, x: list[str], dal: int | None = None, al: int | None = None,
          trimming: float = 0.15) -> dict:
    """Test sup-F di Quandt e Andrews: rottura strutturale a data ignota, cercata nella parte centrale."""
    d = df.loc[dal:al, [y, *x]].dropna()
    n, k = len(d), len(x) + 1
    anni = d.index.to_numpy()
    primo, ultimo = int(np.ceil(trimming * n)), int(np.floor((1 - trimming) * n))
    serie = {}
    for i in range(max(primo, k + 1), min(ultimo, n - k - 1) + 1):
        serie[int(anni[i])] = chow(d, y, x, int(anni[i]))["F"]
    s = pd.Series(serie)
    crit = _ANDREWS_15.get(k)
    return {"anno": int(s.idxmax()), "sup-F": float(s.max()), "critico 10%": crit[0] if crit else np.nan,
            "critico 5%": crit[1] if crit else np.nan, "critico 1%": crit[2] if crit else np.nan, "serie": s}


def differenza_medie(a, b) -> dict:
    """Test t di Welch sulla differenza tra le medie di due gruppi."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    t, p = stats.ttest_ind(b, a, equal_var=False)
    return {"media prima": a.mean(), "media dopo": b.mean(), "differenza": b.mean() - a.mean(), "t": float(t),
            "p-value": float(p), "n prima": len(a), "n dopo": len(b)}


# ---------------------------------------------------------------------------- analisi complete

CONTROLLI = ["treasury_10a", "ln_cape", "vix", "sp500_anno_prima"]


@dataclass
class AnalisiEmpirica:
    dati: pd.DataFrame
    tesi: list[Regressione]
    multivariata: Regressione
    multivariata_dummy: Regressione
    f_controlli: dict
    f_mercati_privati: dict
    chow_2008: dict
    rottura: dict
    eta: dict
    eta_trend: Regressione
    eta_vc_tech: dict
    underpricing: Regressione
    underpricing_f: dict
    europa_quota_ipo: Regressione
    usa_ipo_per_round: Regressione

    def tabella_tesi(self) -> pd.DataFrame:
        righe = []
        for r in self.tesi:
            b = r.risultato.params[r.x[0]]
            righe.append({"Regressione": r.nome, "Periodo": f"{r.campione[0]}-{r.campione[1]}", "Osservazioni": r.n,
                          "Coefficiente": b, "p-value": r.risultato.pvalues[r.x[0]], "R²": r.r2})
        return pd.DataFrame(righe).set_index("Regressione")


def analizza(df: pd.DataFrame | None = None) -> AnalisiEmpirica:
    df = carica_dati() if df is None else df

    # 1) regressioni semplici come nella tesi: IPO sulle operazioni di PE e VC, ultimi undici anni
    tesi = [
        ols(df, "ipo", ["usa_vc_operazioni"], 2015, 2025, "USA: IPO ~ operazioni di VC", hac=False),
        ols(df, "ipo", ["usa_vc_valore_mld_usd"], 2015, 2025, "USA: IPO ~ capitali investiti dal VC", hac=False),
        ols(df, "ipo_vc", ["usa_vc_operazioni"], 2015, 2025, "USA: IPO con VC ~ operazioni di VC", hac=False),
        ols(df, "ipo", ["usa_pe_operazioni"], 2015, 2025, "USA: IPO ~ operazioni di PE", hac=False),
        ols(df, "eur_pe_ipo_numero", ["eur_pe_operazioni"], 2014, 2024,
            "Europa: IPO con PE ~ operazioni di PE", hac=False),
        ols(df, "eur_pe_ipo_numero", ["eur_pe_valore_mld_eur"], 2014, 2024,
            "Europa: IPO con PE ~ capitali investiti dal PE", hac=False),
    ]

    # 2) multivariata di lungo periodo (1991-2025): numero di IPO e condizioni di mercato
    x = [*CONTROLLI, "quota_sponsor"]
    multi = ols(df, "ln_ipo", x, 1991, 2025, "ln(IPO) ~ tassi + valutazioni + volatilità + rendimento + quota PE/VC")
    multi_d = ols(df, "ln_ipo", [*x, "dopo_2008"], 1991, 2025, "… + dopo il 2008")
    f_contr = multi.test_f(CONTROLLI)
    f_priv = multi.test_f(["quota_sponsor"])
    ch = chow(df, "ln_ipo", x, 2008, 1991, 2025)
    rott = sup_f(df, "ln_ipo", ["treasury_10a", "ln_cape", "vix"], 1991, 2025)

    # 3) le aziende restano private più a lungo?
    d = df.loc[1980:2025]
    eta = differenza_medie(d.loc[:2000, "eta_mediana"], d.loc[2001:, "eta_mediana"])
    df = df.assign(tempo=df.index - 1980.0)
    eta_trend = ols(df, "eta_mediana", ["tempo", "bolla_dotcom"], 1980, 2025, "Età mediana ~ tendenza")
    eta_vc = differenza_medie(d.loc[:2000, "eta_mediana_vc_tech"], d.loc[2001:, "eta_mediana_vc_tech"])

    # 4) underpricing e sostegno di VC/PE
    up = ols(df, "rendimento_primo_giorno_medio", ["quota_vc", "ln_cape", "vix", "bolla_dotcom"], 1990, 2025,
             "Rendimento del primo giorno ~ quota VC + valutazioni + volatilità + bolla")
    up_f = up.test_f(["quota_vc"])

    # 5) la via della borsa per le exit: Europa e Stati Uniti
    eu = ols(df, "eur_quota_ipo_exit_pe", ["tempo"], 2014, 2024, "Europa: quota delle exit di PE via IPO ~ tendenza")
    us = ols(df, "usa_ipo_vc_per_mille_round", ["tempo"], 2015, 2025, "USA: IPO con VC per mille round ~ tendenza")
    return AnalisiEmpirica(df, tesi, multi, multi_d, f_contr, f_priv, ch, rott, eta, eta_trend, eta_vc, up, up_f,
                           eu, us)


# ---------------------------------------------------------------------------- dati per singola IPO


def underpricing_per_gruppo(ipo: pd.DataFrame, gruppo: str = "sostegno", rendimento: str = "primo_giorno"
                            ) -> pd.DataFrame:
    """Underpricing medio per gruppo (per esempio VC, PE, nessuno) su dati di singole IPO, con test di Welch
    di ciascun gruppo rispetto alle IPO senza investitore finanziario (gruppo ``"nessuno"``)."""
    rif = ipo.loc[ipo[gruppo] == "nessuno", rendimento]
    righe = []
    for g, r in ipo.groupby(gruppo)[rendimento]:
        t, p = (stats.ttest_ind(r, rif, equal_var=False) if g != "nessuno" else (np.nan, np.nan))
        righe.append({"Gruppo": g, "IPO": len(r), "Media": r.mean(), "Mediana": r.median(), "t vs nessuno": t,
                      "p-value": p})
    return pd.DataFrame(righe).set_index("Gruppo")


def bhar(ipo: pd.DataFrame, rendimento: str = "rendimento_3a", benchmark: str = "benchmark_3a",
         gruppo: str = "sostegno") -> pd.DataFrame:
    """Rendimento anomalo buy-and-hold e *wealth relative* (1 + R) / (1 + R_bench) per gruppo."""
    d = ipo.assign(bhar=ipo[rendimento] - ipo[benchmark])
    out = d.groupby(gruppo).agg(IPO=("bhar", "size"), Rendimento=(rendimento, "mean"),
                                Benchmark=(benchmark, "mean"), BHAR=("bhar", "mean"))
    out["Wealth relative"] = (1 + out["Rendimento"]) / (1 + out["Benchmark"])
    out["t"] = d.groupby(gruppo)["bhar"].apply(lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))
    return out


def ipo_sintetiche(n: int = 2000, seme: int = 3) -> pd.DataFrame:
    """Campione sintetico di singole IPO per provare le funzioni (non sono dati reali)."""
    rng = np.random.default_rng(seme)
    g = rng.choice(["vc", "pe", "nessuno"], n, p=[0.4, 0.2, 0.4])
    media = np.select([g == "vc", g == "pe"], [0.20, 0.10], 0.14)
    bench = rng.normal(0.30, 0.25, n)
    rend = bench + np.select([g == "vc", g == "pe"], [-0.05, 0.0], -0.20) + rng.normal(0, 0.8, n)
    return pd.DataFrame({"sostegno": g, "primo_giorno": media + rng.normal(0, 0.3, n), "rendimento_3a": rend,
                         "benchmark_3a": bench})
