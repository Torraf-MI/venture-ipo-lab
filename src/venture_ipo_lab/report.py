"""Analisi completa (start-up, portafoglio, fondo, IPO), grafici interattivi plotly e report HTML."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from . import dati, esempi, fondo, ipo, portafoglio, valutazione, waterfall
from .cap_table import CapTable

BLU, ARANCIO, ACQUA, VIOLA = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLU_SCURO, ROSSO, GIALLO, GRIGIO = "#1c5cab", "#e34948", "#e8a317", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORI = [BLU_SCURO, ARANCIO, ACQUA, VIOLA, GIALLO, "#d14d9c", GRIGIO, BLU]
COLORI_REGIONI = {"Stati Uniti": BLU_SCURO, "Europa occidentale": ARANCIO}
COLORI_BACKING = {"VC": ARANCIO, "PE": BLU_SCURO, "Nessuno": GRIGIO}
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
NUMERI_PORTAFOGLIO = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]


def _layout(fig: go.Figure, titolo: str, altezza: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza, template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 60, "r": 20, "t": 60, "b": 50}, separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


@dataclass
class Analisi:
    caso: dict
    cap_table: CapTable
    griglia_exit: pd.DataFrame
    multipli_scenari: dict[str, pd.DataFrame]
    first_chicago: pd.DataFrame
    valore_fc: float
    vc: valutazione.EsitoVC
    sens_vc: pd.DataFrame
    portafogli: pd.DataFrame
    distribuzioni: dict[int, np.ndarray]
    concentrazione: pd.Series
    follow_on: pd.DataFrame
    flussi_fondo: pd.DataFrame
    indicatori_fondo: pd.Series
    scenari_fondo: pd.DataFrame
    piano: pd.DataFrame
    commitment: float
    serie: pd.DataFrame
    campione_ipo: pd.DataFrame
    replica: pd.DataFrame
    multivariate: dict[str, ipo.Regressione]
    test_f: dict[str, pd.Series]
    chow: dict[str, pd.Series]
    ricerca: dict[str, pd.DataFrame]
    eta: pd.DataFrame
    backing: pd.DataFrame
    test_underpricing: pd.DataFrame
    test_bhar: pd.DataFrame
    reg_underpricing: ipo.Regressione
    patrimonio: float
    allocazione: float
    dati_dimostrativi: bool


def analizza(
    caso: dict | None = None,
    serie: pd.DataFrame | None = None,
    campione_ipo: pd.DataFrame | None = None,
    patrimonio: float = 5_000_000,
    allocazione: float = 0.10,
    simulazioni: int = 20_000,
) -> Analisi:
    caso = caso or esempi.CASO_DEMO
    dimostrativi = serie is None
    serie = dati.genera_serie() if serie is None else serie
    if campione_ipo is None:
        campione_ipo = dati.genera_ipo(serie) if dimostrativi else None
    ct = esempi.costruisci_cap_table(caso)
    post = ct.round_info[-1]["Post-money"]
    griglia = waterfall.tabella_exit(ct, np.linspace(0, 6 * post, 121))
    sc = esempi.scenari(caso)
    multipli = {s.nome: waterfall.multipli_exit(ct, s.valore_exit) for s in sc if s.valore_exit > 0}
    fc, valore_fc = valutazione.first_chicago(sc, caso["exit"]["tasso"])
    v = caso["valutazione_vc"]
    vc = valutazione.metodo_vc(v["valore_exit"], v["investimento"], v.get("multiplo_target"), v.get("tir_target"),
                               v.get("anni"), v.get("diluizioni_future"))
    sens = valutazione.sensitivita_vc(v["valore_exit"], v["investimento"], [5, 7.5, 10, 15, 20],
                                      [1.0, 0.85, 0.7, 0.55])

    distr = portafoglio.DistribuzioneRendimenti()
    port = portafoglio.simula_portafogli(NUMERI_PORTAFOGLIO, distr, simulazioni)
    distribuzioni = {n: portafoglio.distribuzione_portafoglio(n, distr, simulazioni) for n in (5, 20, 50)}
    follow = pd.DataFrame({f"Step-up {s:g}x": portafoglio.con_follow_on(20, 0.4, distr, step_up=s,
                                                                         simulazioni=simulazioni)
                           for s in (1.5, 2.5, 4.0)}).T

    flussi = fondo.flussi_fondo(1.0)
    rng = np.random.default_rng(3)
    indice = pd.Series(np.cumprod(1 + rng.normal(0.08, 0.16, len(flussi))), index=flussi.index)
    ind = fondo.indicatori(flussi, indice)
    sc_fondo = fondo.scenari_fondo({"Quartile inferiore": 0.02, "Mediana": 0.09, "Quartile superiore": 0.17,
                                    "Primo decile": 0.26})
    piano, c0 = fondo.piano_commitment(patrimonio, allocazione)

    regioni = list(serie["regione"].unique())
    prep = {r: ipo.prepara(serie, r) for r in regioni}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multiv = {r: ipo.multivariata(prep[r]) for r in regioni}
        test_f = {r: ipo.ftest_privato(prep[r]) for r in regioni}
        chow = {r: ipo.chow(prep[r]) for r in regioni}
        ricerca = {r: ipo.ricerca_rottura(prep[r]) for r in regioni}
    return Analisi(
        caso, ct, griglia, multipli, fc, valore_fc, vc, sens, port, distribuzioni,
        portafoglio.concentrazione(distr), follow, flussi, ind, sc_fondo, piano, c0, serie, campione_ipo,
        ipo.replica_tesi(serie), multiv, test_f, chow, ricerca, ipo.eta_alla_quotazione(serie),
        ipo.confronto_backing(campione_ipo) if campione_ipo is not None else None,
        ipo.ttest_backing(campione_ipo) if campione_ipo is not None else None,
        ipo.ttest_backing(campione_ipo, "bhar") if campione_ipo is not None else None,
        ipo.regressione_underpricing(campione_ipo) if campione_ipo is not None else None,
        patrimonio, allocazione, dimostrativi,
    )


# --------------------------------------------------------------------------- #
# Grafici: start-up
# --------------------------------------------------------------------------- #
def grafico_diluizione(a: Analisi) -> go.Figure:
    d = a.cap_table.diluizione()
    ordine = [c for c in d.columns if d[c].iloc[0] > 0] + [c for c in d.columns if d[c].iloc[0] == 0]
    fig = go.Figure()
    for i, c in enumerate(ordine):
        fig.add_bar(x=d.index, y=d[c], name=c, marker_color=COLORI[i % len(COLORI)],
                    hovertemplate="%{x}<br>" + c + ": %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%", title="Quota fully diluted")
    return _layout(fig, "Chi possiede la società dopo ogni round")


def grafico_waterfall(a: Analisi) -> go.Figure:
    g = a.griglia_exit / 1e6
    x = g.index / 1e6
    fig = go.Figure()
    for i, c in enumerate(g.columns):
        fig.add_scatter(x=x, y=g[c], name=c, mode="lines", line={"width": 2.2, "color": COLORI[i % len(COLORI)]},
                        hovertemplate="Exit %{x:,.0f} mln<br>" + c + ": %{y:,.1f} mln<extra></extra>")
    for s in esempi.scenari(a.caso):
        if 0 < s.valore_exit <= x.max() * 1e6:
            fig.add_vline(x=s.valore_exit / 1e6, line_dash="dot", line_color=GRIGIO,
                          annotation_text=s.nome, annotation_font_size=10)
    fig.update_xaxes(title="Valore di exit (milioni)")
    fig.update_yaxes(title="Proventi (milioni)")
    return _layout(fig, "Quanto riceve ciascuno al variare del prezzo di vendita", 460)


def grafico_quota_proventi(a: Analisi) -> go.Figure:
    g = a.griglia_exit.iloc[1:]
    quota = g.div(g.index.to_numpy(), axis=0)
    x = g.index / 1e6
    fondatori = [c for c in quota.columns if c in a.caso["fondatori"]]
    azioni = a.cap_table.quote_per_titolare()
    fig = go.Figure()
    for i, c in enumerate(fondatori):
        fig.add_scatter(x=x, y=quota[c], name=f"{c}: quota dei proventi", line={"color": COLORI[i], "width": 2.4})
        fig.add_scatter(x=[x.min(), x.max()], y=[azioni[c]] * 2, name=f"{c}: quota azionaria",
                        line={"color": COLORI[i], "dash": "dash", "width": 1.4})
    fig.update_yaxes(tickformat=".0%", title="Quota")
    fig.update_xaxes(title="Valore di exit (milioni)")
    return _layout(fig, "Fondatori: la quota azionaria non è la quota dei proventi")


# --------------------------------------------------------------------------- #
# Grafici: portafoglio e fondo
# --------------------------------------------------------------------------- #
def grafico_portafogli(a: Analisi) -> go.Figure:
    p = a.portafogli
    fig = go.Figure()
    fig.add_scatter(x=p.index, y=p["Prob. di perdere capitale"], name="Probabilità di perdere capitale",
                    mode="lines+markers", line={"color": ROSSO, "width": 2.4})
    fig.add_scatter(x=p.index, y=p["Prob. almeno 2x"], name="Probabilità di almeno 2x", mode="lines+markers",
                    line={"color": ACQUA, "width": 2.4})
    fig.add_scatter(x=p.index, y=p["Prob. di una fund returner"], name="Probabilità di una fund returner",
                    mode="lines+markers", line={"color": VIOLA, "width": 2, "dash": "dot"})
    fig.update_xaxes(type="log", title="Numero di start-up in portafoglio", tickvals=NUMERI_PORTAFOGLIO)
    fig.update_yaxes(tickformat=".0%", title="Probabilità", range=[0, 0.6])
    return _layout(fig, "Quante start-up servono: rischio di perdita e numero di partecipazioni")


def grafico_distribuzioni(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for i, (n, v) in enumerate(a.distribuzioni.items()):
        fig.add_histogram(x=np.clip(v, 0, 4), name=f"{n} start-up", opacity=0.55, nbinsx=80,
                          histnorm="probability", marker_color=[ROSSO, BLU_SCURO, ACQUA][i])
    fig.add_vline(x=1, line_dash="dash", line_color=TESTO)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Multiplo del portafoglio (valori oltre 4x raggruppati a 4x)")
    fig.update_yaxes(title="Frequenza", tickformat=".0%")
    return _layout(fig, "Stesso rendimento atteso, rischio molto diverso")


def grafico_concentrazione(a: Analisi) -> go.Figure:
    c = a.concentrazione
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=BLU_SCURO, text=[f"{v:.0%}" for v in c.values],
                           textposition="outside"))
    fig.update_yaxes(tickformat=".0%", title="Quota dei proventi totali", range=[0, 1])
    return _layout(fig, "La legge di potenza: pochi investimenti fanno quasi tutto il risultato", 380)


def grafico_jcurve(a: Analisi) -> go.Figure:
    f = a.flussi_fondo
    fig = go.Figure()
    fig.add_bar(x=f.index, y=-f["Richiami"], name="Richiami", marker_color=ROSSO)
    fig.add_bar(x=f.index, y=f["Distribuzioni"], name="Distribuzioni", marker_color=ACQUA)
    fig.add_scatter(x=f.index, y=f["Flusso cumulato"], name="Flusso netto cumulato (J-curve)",
                    line={"color": TESTO, "width": 2.6})
    fig.add_scatter(x=f.index, y=f["NAV"], name="NAV", line={"color": BLU_SCURO, "dash": "dot", "width": 2})
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title="Anno di vita del fondo", dtick=1)
    fig.update_yaxes(title="Per euro di commitment", tickformat=".2f")
    return _layout(fig, "Un fondo di venture capital visto dall'investitore: la J-curve", 440)


def grafico_piano(a: Analisi) -> go.Figure:
    p = a.piano
    fig = go.Figure()
    fig.add_scatter(x=p.index, y=p["Quota NAV"], name="NAV sul patrimonio", line={"color": BLU_SCURO, "width": 2.6})
    fig.add_scatter(x=p.index, y=p["Quota NAV + impegni"], name="NAV + impegni non richiamati",
                    line={"color": ARANCIO, "width": 2, "dash": "dash"})
    fig.add_hline(y=a.allocazione, line_dash="dot", line_color=GRIGIO, annotation_text="Obiettivo")
    fig.update_xaxes(title="Anno", dtick=1)
    fig.update_yaxes(tickformat=".0%", title="Quota del patrimonio")
    return _layout(fig, "Piano di commitment: quanto tempo serve per arrivare all'allocazione obiettivo")


# --------------------------------------------------------------------------- #
# Grafici: IPO (tesi)
# --------------------------------------------------------------------------- #
def grafico_ipo_privato(a: Analisi) -> go.Figure:
    regioni = list(a.serie["regione"].unique())
    fig = make_subplots(rows=1, cols=len(regioni), subplot_titles=regioni,
                        specs=[[{"secondary_y": True}] * len(regioni)])
    for j, r in enumerate(regioni, start=1):
        d = a.serie[a.serie["regione"] == r]
        fig.add_bar(x=d["anno"], y=d["ipo"], name="Numero di IPO", marker_color=COLORI_REGIONI.get(r, BLU),
                    opacity=0.75, showlegend=j == 1, row=1, col=j)
        fig.add_scatter(x=d["anno"], y=d["capitale_privato"], name="Capitale VC e PE investito (mld)",
                        line={"color": TESTO, "width": 2.2}, showlegend=j == 1, row=1, col=j, secondary_y=True)
        fig.add_vline(x=dati.ANNO_ROTTURA - 0.5, line_dash="dot", line_color=ROSSO, row=1, col=j)
    fig.update_yaxes(title_text="IPO", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Capitale privato (mld)", secondary_y=True, showgrid=False, rangemode="tozero")
    return _layout(fig, "IPO e capitale privato, 1980-2025: dal 2008 le due serie si separano", 440)


def grafico_replica(a: Analisi) -> go.Figure:
    r = a.replica
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(r["Modello"].unique()))
    for j, m in enumerate(r["Modello"].unique(), start=1):
        d = r[r["Modello"] == m]
        for f, col in zip(d["Finestra"].unique(), [GRIGIO, BLU_SCURO]):
            dd = d[d["Finestra"] == f]
            fig.add_bar(x=dd["Regione"], y=dd["R²"], name=f"Finestra {f}", marker_color=col, showlegend=j == 1,
                        text=[f"{v:.2f}".replace(".", ",") for v in dd["R²"]], textposition="outside", row=1, col=j)
    fig.update_yaxes(range=[0, 1.05], title_text="R²", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    return _layout(fig, "Replica della tesi: regressioni semplici su 20 e su 10 anni", 400)


def grafico_effetto(a: Analisi) -> go.Figure:
    regioni = list(a.test_f)
    prima = [a.test_f[r]["Effetto prima del 2008"] for r in regioni]
    dopo = [a.test_f[r]["Effetto dopo il 2008"] for r in regioni]
    fig = go.Figure()
    fig.add_bar(x=regioni, y=prima, name="Fino al 2007", marker_color=GRIGIO,
                text=[f"{v:+.2f}".replace(".", ",") for v in prima], textposition="outside")
    fig.add_bar(x=regioni, y=dopo, name="Dal 2008", marker_color=ROSSO,
                text=[f"{v:+.2f}".replace(".", ",") for v in dopo], textposition="outside")
    fig.add_hline(y=0, line_color=TESTO)
    fig.update_yaxes(title="Elasticità delle IPO al capitale privato")
    return _layout(fig, "A parità di borsa, volatilità e tassi: più capitale privato, meno IPO dal 2008", 400)


def grafico_chow(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for r, d in a.ricerca.items():
        fig.add_scatter(x=d.index, y=d["F di Chow"], name=r, mode="lines+markers",
                        line={"color": COLORI_REGIONI.get(r, BLU), "width": 2.2})
    k = len(ipo.CONTROLLI) + 2
    n = len(a.serie[a.serie["regione"] == a.serie["regione"].iloc[0]].query("anno >= 1985"))
    from scipy import stats
    fig.add_hline(y=stats.f.ppf(0.95, k, n - 2 * k), line_dash="dash", line_color=GRIGIO,
                  annotation_text="Soglia al 5%")
    fig.update_xaxes(title="Anno di rottura ipotizzato", dtick=2)
    fig.update_yaxes(title="F di Chow")
    return _layout(fig, "Test di Chow: in che anno cambia il legame tra IPO e mercato", 400)


def grafico_eta(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for r in a.serie["regione"].unique():
        d = a.serie[a.serie["regione"] == r]
        fig.add_scatter(x=d["anno"], y=d["eta_mediana_ipo"], name=r, mode="lines",
                        line={"color": COLORI_REGIONI.get(r, BLU), "width": 2.4})
    fig.update_yaxes(title="Età mediana alla quotazione (anni)")
    return _layout(fig, "Le aziende arrivano in borsa sempre più tardi", 380)


def grafico_backing(a: Analisi) -> go.Figure:
    b = a.backing.reset_index()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Underpricing medio del primo giorno",
                                                        "Rendimento a 3 anni rispetto al mercato"])
    for k in ("VC", "PE", "Nessuno"):
        d = b[b["backing"] == k]
        nome = {"VC": "Con venture capital", "PE": "Con private equity", "Nessuno": "Senza VC o PE"}[k]
        fig.add_bar(x=d["regione"], y=d["Underpricing medio"], name=nome, marker_color=COLORI_BACKING[k], row=1, col=1)
        fig.add_bar(x=d["regione"], y=d["Rendimento in eccesso a 3 anni"], name=nome, marker_color=COLORI_BACKING[k],
                    showlegend=False, row=1, col=2)
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "IPO con e senza investitori professionali alle spalle", 420)


def _disponibili(a: Analisi) -> dict:
    g = {"diluizione": grafico_diluizione, "waterfall": grafico_waterfall, "quota_proventi": grafico_quota_proventi,
         "portafogli": grafico_portafogli, "distribuzioni": grafico_distribuzioni,
         "concentrazione": grafico_concentrazione, "jcurve": grafico_jcurve, "piano": grafico_piano,
         "ipo_privato": grafico_ipo_privato, "replica": grafico_replica, "effetto": grafico_effetto,
         "chow": grafico_chow, "eta": grafico_eta}
    if a.backing is not None:
        g["backing"] = grafico_backing
    return g


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede ``kaleido`` e un browser Chrome/Chromium)."""
    out = Path(cartella)
    out.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, f in _disponibili(a).items():
        percorso = out / f"{nome}.png"
        f(a).write_image(percorso, width=900, scale=scala)
        percorsi.append(percorso)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float, decimali: int = 1, segno: bool = False) -> str:
    if x is None or pd.isna(x):
        return "–"
    testo = f"{x:+.{decimali}%}" if segno else f"{x:.{decimali}%}"
    return testo.replace(".", ",").replace("-", "−")


def _num(x: float, decimali: int = 2, segno: bool = False) -> str:
    if x is None or pd.isna(x):
        return "–"
    testo = f"{x:+,.{decimali}f}" if segno else f"{x:,.{decimali}f}"
    return testo.replace(",", "§").replace(".", ",").replace("§", ".").replace("-", "−")


def _eur(x: float) -> str:
    if x is None or pd.isna(x):
        return "–"
    if abs(x) >= 1e6:
        return f"{_num(x / 1e6, 1)} mln €"
    return f"{_num(x, 0)} €"


def _p(x: float) -> str:
    return "< 0,001" if x < 0.001 else _num(x, 3)


def contesto(a: Analisi) -> dict:
    ct = a.cap_table
    quote = ct.quote_per_titolare()
    fondatori = list(a.caso["fondatori"])
    q_fond = sum(quote[f] for f in fondatori)
    rounds = [{"nome": n, "pre": _eur(r["Pre-money"]), "inv": _eur(r["Investimento"]),
               "safe": _eur(r["SAFE convertiti"]) if r["SAFE convertiti"] else "–", "post": _eur(r["Post-money"]),
               "prezzo": _num(r["Prezzo per azione"], 3) + " €",
               "pool": _eur(r["Pre-money"] - r["Pre-money effettivo"]),
               "eff": _eur(r["Pre-money effettivo"])}
              for n, r in ct.rounds().iterrows()]
    libro = ct.tabella().groupby("Titolare").agg(Quota=("Quota", "sum"), Investito=("Investito", "sum"))
    libro = [{"nome": t, "quota": _pct(r["Quota"]), "inv": _eur(r["Investito"]) if r["Investito"] > 0 else "–"}
             for t, r in libro.sort_values("Quota", ascending=False).iterrows()]
    classi = [{"nome": c.nome, "pref": f"{_num(c.multiplo_pref, 1)}x" if c.privilegiata else "–",
               "tipo": ("partecipante"
                        + (f" con tetto {_num(c.cap_partecipazione, 1)}x" if c.cap_partecipazione else ""))
               if c.partecipante else ("non partecipante" if c.privilegiata else "ordinarie"),
               "seniority": c.seniority if c.privilegiata else "–", "inv": _eur(c.investito) if c.investito else "–",
               "soglia": _eur(waterfall.soglia_conversione(ct, c.nome)) if waterfall.soglia_conversione(ct, c.nome)
               else "–"} for c in ct.classi.values()]
    investitore = a.caso.get("investitore")
    scen = []
    for s in esempi.scenari(a.caso):
        m = a.multipli_scenari.get(s.nome)
        riga = {"nome": s.nome, "prob": _pct(s.probabilita, 0), "valore": _eur(s.valore_exit), "anni": s.anni,
                "fond": _eur(sum(m.loc[f, "Proventi"] for f in fondatori)) if m is not None else "0 €",
                "fond_q": _pct(sum(m.loc[f, "Quota dei proventi"] for f in fondatori), 0) if m is not None else "–",
                "inv": _num(m.loc[investitore, "Multiplo"], 1) + "x" if m is not None and investitore in m.index
                else "0x"}
        scen.append(riga)
    fc = [{"nome": n, "prob": _pct(r["Probabilità"], 0), "valore": _eur(r["Valore di exit"]), "anni": int(r["Anni"]),
           "vp": _eur(r["Valore attuale"]), "contr": _eur(r["Contributo"])} for n, r in a.first_chicago.iterrows()]
    sens = {"colonne": [_pct(c, 0) for c in a.sens_vc.columns],
            "righe": [{"m": f"{_num(m, 1)}x", "v": [_eur(x) if x > 0 else "non sostenibile" for x in r]}
                      for m, r in a.sens_vc.iterrows()]}

    p = a.portafogli
    port = [{"n": n, "media": _num(r["Multiplo medio"]) + "x", "mediana": _num(r["Multiplo mediano"]) + "x",
             "perdita": _pct(r["Prob. di perdere capitale"], 0), "due": _pct(r["Prob. almeno 2x"], 0),
             "p5": _num(r["5° percentile"]) + "x", "p95": _num(r["95° percentile"]) + "x"}
            for n, r in p.iterrows()]
    follow = [{"nome": n, "med_senza": _num(r["Multiplo mediano senza riserva"]) + "x",
               "med_con": _num(r["Multiplo mediano con riserva"]) + "x",
               "media_con": _num(r["Multiplo medio con riserva"]) + "x",
               "perd_con": _pct(r["Prob. perdita con riserva"], 0)} for n, r in a.follow_on.iterrows()]
    fs = [{"nome": n, "tvpi": _num(r["TVPI"]) + "x", "dpi": _num(r["DPI"]) + "x", "tir": _pct(r["TIR"])}
          for n, r in a.scenari_fondo.iterrows()]
    i = a.indicatori_fondo
    pareggio = fondo.anno_pareggio(a.flussi_fondo)
    minimo = a.flussi_fondo["Flusso cumulato"].min()
    anno_target = next((int(k) for k, v in a.piano["Quota NAV"].items() if v >= 0.95 * a.allocazione), None)

    rep = [{"reg": r["Regione"], "fin": r["Finestra"], "mod": r["Modello"], "n": r["Osservazioni"],
            "pend": _num(r["Pendenza"], 3), "p": _p(r["p-value"]), "r2": _num(r["R²"])}
           for _, r in a.replica.iterrows()]
    multiv = {reg: [{"nome": k, "coef": _num(v["Coefficiente"], 3), "t": _num(v["t"]), "p": _p(v["p-value"]),
                     "sig": v["p-value"] < 0.05} for k, v in m.tabella().iterrows()]
              for reg, m in a.multivariate.items()}
    r2 = {reg: _num(m.r2_corretto) for reg, m in a.multivariate.items()}
    testf = [{"reg": reg, "f": _num(t["F (privato conta)"]), "p": _p(t["p-value (privato conta)"]),
              "r2_con": _num(t["R² con privato"]), "r2_senza": _num(t["R² senza privato"]),
              "f2": _num(t["F (effetto cambia dopo il 2008)"]), "p2": _p(t["p-value (effetto cambia)"]),
              "prima": _num(t["Effetto prima del 2008"], 2, True), "dopo": _num(t["Effetto dopo il 2008"], 2, True),
              "chow": _num(a.chow[reg]["F di Chow"]), "chow_p": _p(a.chow[reg]["p-value"]),
              "max": int(a.ricerca[reg]["F di Chow"].idxmax())}
             for reg, t in a.test_f.items()]
    eta = [{"reg": reg, "dec": _num(r["Anni in più per decennio"], 1, True),
            "prima": _num(r["Età media prima del 2008"], 1),
            "dopo": _num(r["Età media dal 2008"], 1)} for reg, r in a.eta.iterrows()]
    back = None
    if a.backing is not None:
        back = {
            "righe": [{"reg": reg, "b": {"VC": "Venture capital", "PE": "Private equity", "Nessuno": "Nessuno"}[b],
                       "n": int(r["IPO"]), "up": _pct(r["Underpricing medio"]),
                       "bhar": _pct(r["Rendimento in eccesso a 3 anni"], 1, True),
                       "sotto": _pct(r["Quota sotto il mercato"], 0), "eta": _num(r["Età mediana"], 1)}
                      for (reg, b), r in a.backing.iterrows()],
            "test": [{"reg": r["Regione"], "conf": r["Confronto"], "d_up": _pct(r["Differenza"], 1, True),
                      "p_up": _p(r["p-value t"]), "d_bhar": _pct(b["Differenza"], 1, True),
                      "p_bhar": _p(b["p-value t"])}
                     for (_, r), (_, b) in zip(a.test_underpricing.iterrows(), a.test_bhar.iterrows())],
            "reg": [{"nome": {"const": "Costante", "VC": "Con venture capital", "PE": "Con private equity",
                              "log_eta": "Età (log)", "log_raccolta": "Raccolta (log)", "Europa": "Europa"}[k],
                     "coef": _pct(v["Coefficiente"], 1, True), "t": _num(v["t"]), "p": _p(v["p-value"])}
                    for k, v in a.reg_underpricing.tabella().rename(
                        index={v: k for k, v in ipo.NOMI.items()}).iterrows()],
            "n": len(a.campione_ipo),
        }

    us = a.test_f.get("Stati Uniti", next(iter(a.test_f.values())))
    kpi = [
        ("Quota dei fondatori dopo la Serie B" if "Serie B" in ct.classi else "Quota dei fondatori", _pct(q_fond, 0)),
        ("Valore First Chicago (post-money)", _eur(a.valore_fc)),
        ("Prob. di perdita con 10 start-up", _pct(p.loc[10, "Prob. di perdere capitale"], 0)),
        ("Prob. di perdita con 50 start-up", _pct(p.loc[50, "Prob. di perdere capitale"], 0)),
        ("Fondo: la J-curve torna positiva", f"anno {pareggio}" if pareggio else "–"),
        ("Effetto del capitale privato sulle IPO dal 2008", _num(us["Effetto dopo il 2008"], 2, True)),
    ]
    return dict(
        nome=a.caso["nome"], today=date.today().strftime("%d/%m/%Y"), kpi=kpi, rounds=rounds, libro=libro,
        classi=classi, scenari=scen, investitore=investitore, fc=fc, valore_fc=_eur(a.valore_fc),
        tasso_fc=_pct(a.caso["exit"]["tasso"], 0), q_fond=_pct(q_fond, 0),
        vc={"exit": _eur(a.vc.valore_exit), "mult": _num(a.vc.multiplo_target, 1) + "x",
            "rit": _pct(a.vc.ritenzione, 0), "post": _eur(a.vc.post_money), "pre": _eur(a.vc.pre_money),
            "quota": _pct(a.vc.quota_richiesta), "quota_exit": _pct(a.vc.quota_all_exit),
            "inv": _eur(a.caso["valutazione_vc"]["investimento"])},
        sens=sens, port=port, follow=follow, conc={k: _pct(v, 0) for k, v in a.concentrazione.items()},
        media_singola=_num(portafoglio.DistribuzioneRendimenti().media()) + "x",
        fondo={"tvpi": _num(i["TVPI"]) + "x", "tir": _pct(i["TIR"]), "pme": _num(i["KS-PME"]),
               "pareggio": pareggio, "minimo": _pct(-minimo, 0)},
        fondi_scenari=fs,
        piano={"c0": _eur(a.commitment), "pat": _eur(a.patrimonio), "obiettivo": _pct(a.allocazione, 0),
                                 "anno": anno_target,
                                 "impegni": _pct(a.piano["Quota NAV + impegni"].iloc[-1], 0)},
        replica=rep, multiv=multiv, r2=r2, testf=testf, eta=eta, back=back, demo=a.dati_dimostrativi,
        periodo=f"{int(a.serie['anno'].min())}-{int(a.serie['anno'].max())}",
    )


def build_report(a: Analisi, output: str | Path, offline: bool = False) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    config = {"displaylogo": False, "responsive": True}
    grafici = {n: f(a).to_html(full_html=False, include_plotlyjs=False, config=config)
               for n, f in _disponibili(a).items()}
    env = Environment(loader=PackageLoader("venture_ipo_lab", "templates"), autoescape=select_autoescape(["html"]))
    html = env.get_template("report.html.j2").render(
        **contesto(a), grafici=grafici, plotly_cdn=PLOTLY_CDN, plotly_inline=get_plotlyjs() if offline else None)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
