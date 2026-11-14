"""Un fondo di venture capital visto dall'investitore (LP): flussi, J-curve, indicatori e piano di commitment.

I flussi seguono il modello di **Takahashi e Alexander** (Yale, 2002), lo standard per pianificare i richiami e le
distribuzioni di un fondo chiuso:

* richiamo nell'anno t: ``RC_t × (commitment − richiamato finora)``;
* distribuzione: ``RD_t × NAV_{t−1} × (1 + G)``, con ``RD_t = (t / L)^B``;
* NAV: ``NAV_{t−1} × (1 + G) + richiamo − distribuzione``.

``G`` è la crescita annua del valore, ``L`` la vita del fondo, ``B`` (bow) controlla quanto le distribuzioni
sono rimandate verso la fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass
class ParametriFondo:
    vita: int = 12
    crescita: float = 0.14  # rendimento annuo lordo del NAV
    bow: float = 2.8
    richiami: list[float] = field(default_factory=lambda: [0.25, 0.33, 0.40, 0.50, 0.60])  # poi 1

    def tasso_richiamo(self, t: int) -> float:
        return self.richiami[t - 1] if t <= len(self.richiami) else 1.0


def flussi_fondo(commitment: float = 1.0, p: ParametriFondo | None = None) -> pd.DataFrame:
    """Richiami, distribuzioni e NAV anno per anno di un fondo (anni 1..vita)."""
    p = p or ParametriFondo()
    nav, richiamato = 0.0, 0.0
    righe = []
    for t in range(1, p.vita + 1):
        rc = p.tasso_richiamo(t) * (commitment - richiamato)
        rd = 1.0 if t == p.vita else (t / p.vita) ** p.bow
        dist = rd * nav * (1 + p.crescita)
        nav = nav * (1 + p.crescita) + rc - dist
        richiamato += rc
        righe.append({"Anno": t, "Richiami": rc, "Distribuzioni": dist, "NAV": nav})
    df = pd.DataFrame(righe).set_index("Anno")
    df["Flusso netto"] = df["Distribuzioni"] - df["Richiami"]
    df["Flusso cumulato"] = df["Flusso netto"].cumsum()  # la J-curve
    return df


def tir(flussi: np.ndarray | list[float]) -> float:
    """TIR annuo di una serie di flussi annuali (negativi = versamenti)."""
    f = np.asarray(flussi, dtype=float)
    van = lambda r: float(np.sum(f / (1 + r) ** np.arange(len(f))))  # noqa: E731
    return brentq(van, -0.99, 10.0)


def indicatori(df: pd.DataFrame, indice: pd.Series | None = None) -> pd.Series:
    """TVPI, DPI, RVPI, TIR e, se è dato un indice di borsa, la PME di Kaplan e Schoar (2005).

    ``indice`` è il livello dell'indice a fine anno, sugli stessi anni del fondo. KS-PME > 1 vuol dire che il
    fondo ha battuto la borsa a parità di tempi dei flussi.
    """
    versato = df["Richiami"].sum()
    distribuito = df["Distribuzioni"].sum()
    nav = df["NAV"].iloc[-1]
    flussi = (df["Distribuzioni"] - df["Richiami"]).to_numpy().copy()
    flussi[-1] += nav
    out = {
        "Versato": versato, "Distribuito": distribuito, "NAV finale": nav,
        "TVPI": (distribuito + nav) / versato, "DPI": distribuito / versato, "RVPI": nav / versato,
        "TIR": tir(flussi),
    }
    if indice is not None:
        i = indice.reindex(df.index).to_numpy(dtype=float)
        fine = i[-1]
        pv_dist = np.sum(df["Distribuzioni"].to_numpy() * fine / i) + nav
        pv_rich = np.sum(df["Richiami"].to_numpy() * fine / i)
        out["KS-PME"] = pv_dist / pv_rich
    return pd.Series(out)


def anno_pareggio(df: pd.DataFrame) -> int | None:
    """Primo anno in cui il flusso cumulato torna positivo (fine della J-curve)."""
    pos = df.index[df["Flusso cumulato"] >= 0]
    return int(pos[0]) if len(pos) else None


def piano_commitment(
    patrimonio: float,
    allocazione_target: float,
    anni: int = 15,
    p: ParametriFondo | None = None,
    crescita_patrimonio: float = 0.04,
) -> tuple[pd.DataFrame, float]:
    """Commitment annuo costante che porta il NAV a regime sulla quota target del patrimonio.

    A regime, con un nuovo fondo ogni anno, il NAV totale per euro di commitment annuo è la somma dei NAV di un
    fondo lungo tutta la sua vita. Il commitment cresce con il patrimonio. Restituisce la traiettoria anno per anno
    (NAV, impegni non richiamati, quota sul patrimonio) e il commitment del primo anno.
    """
    p = p or ParametriFondo()
    profilo = flussi_fondo(1.0, p)
    # NAV a regime per euro di commitment dell'anno in corso: i vintage più vecchi pesano meno, perché
    # sono stati impegnati quando il patrimonio era più piccolo.
    sconto = (1 + crescita_patrimonio) ** -(profilo.index.to_numpy() - 1)
    nav_per_euro = float((profilo["NAV"].to_numpy() * sconto).sum())
    c0 = allocazione_target * patrimonio / nav_per_euro
    righe = []
    for anno in range(1, anni + 1):
        pat = patrimonio * (1 + crescita_patrimonio) ** (anno - 1)
        nav = richiami = dist = non_richiamato = 0.0
        for v in range(1, anno + 1):  # vintage v, età anno - v + 1
            eta = anno - v + 1
            c = c0 * (1 + crescita_patrimonio) ** (v - 1)
            if eta <= p.vita:
                r = profilo.loc[eta]
                nav += c * r["NAV"]
                richiami += c * r["Richiami"]
                dist += c * r["Distribuzioni"]
                non_richiamato += c * (1 - profilo["Richiami"].loc[:eta].sum())
        righe.append({"Anno": anno, "Patrimonio": pat, "Commitment": c0 * (1 + crescita_patrimonio) ** (anno - 1),
                      "Richiami": richiami, "Distribuzioni": dist, "NAV": nav, "Non richiamato": non_richiamato,
                      "Quota NAV": nav / pat, "Quota NAV + impegni": (nav + non_richiamato) / pat})
    return pd.DataFrame(righe).set_index("Anno"), c0


def scenari_fondo(scenari: dict[str, float], p: ParametriFondo | None = None) -> pd.DataFrame:
    """Indicatori del fondo per diverse crescite annue del NAV (es. quartili di performance)."""
    p = p or ParametriFondo()
    righe = {}
    for nome, g in scenari.items():
        q = ParametriFondo(p.vita, g, p.bow, list(p.richiami))
        righe[nome] = indicatori(flussi_fondo(1.0, q))
    return pd.DataFrame(righe).T
