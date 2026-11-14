"""Portafogli di investimenti angel e venture: rendimenti a legge di potenza e numero di start-up necessario.

Il rendimento di una singola start-up si modella come una mistura:

* con probabilità ``prob_perdita`` l'investimento rende meno di 1x (spesso zero): multiplo ~ Beta(a, b) su [0, 1);
* altrimenti il multiplo segue una **Pareto troncata** con minimo 1x, indice di coda ``alfa`` e massimo ``tetto``.

Con ``alfa`` tra 1,5 e 2 la coda è abbastanza spessa da far dipendere il risultato di un portafoglio da pochissimi
casi: è la ragione per cui un angel con 5 partecipazioni e uno con 50 hanno rischi molto diversi anche se il
rendimento atteso è identico. I parametri di default producono distribuzioni dello stesso ordine di grandezza di
quelle riportate negli studi pubblici sugli angel investor; si possono cambiare per fare analisi di sensitività.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DistribuzioneRendimenti:
    prob_perdita: float = 0.55
    perdita_a: float = 0.6  # Beta(a, b): con a < 1 molti esiti sono vicini a zero
    perdita_b: float = 1.6
    alfa: float = 1.75
    tetto: float = 250.0

    def campiona(self, n: int | tuple, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(n)
        perdita = rng.beta(self.perdita_a, self.perdita_b, n)
        # Pareto troncata su [1, tetto] con il metodo dell'inversa.
        v = rng.random(n)
        h = self.tetto ** (-self.alfa)
        coda = (1 - v * (1 - h)) ** (-1 / self.alfa)
        return np.where(u < self.prob_perdita, perdita, coda)

    def media(self) -> float:
        m_perdita = self.perdita_a / (self.perdita_a + self.perdita_b)
        a, t = self.alfa, self.tetto
        h = t ** (-a)
        m_coda = a / (a - 1) * (1 - t ** (1 - a)) / (1 - h) if a != 1 else np.log(t) / (1 - h)
        return self.prob_perdita * m_perdita + (1 - self.prob_perdita) * m_coda


def simula_portafogli(
    numeri: list[int],
    distr: DistribuzioneRendimenti | None = None,
    simulazioni: int = 20_000,
    seme: int = 7,
) -> pd.DataFrame:
    """Statistiche del multiplo di portafoglio (pesi uguali) per ogni numero di start-up."""
    distr = distr or DistribuzioneRendimenti()
    rng = np.random.default_rng(seme)
    righe = []
    for n in numeri:
        m = distr.campiona((simulazioni, n), rng)
        port = m.mean(axis=1)
        righe.append({
            "Start-up": n,
            "Multiplo medio": port.mean(),
            "Multiplo mediano": np.median(port),
            "Prob. di perdere capitale": (port < 1).mean(),
            "Prob. almeno 2x": (port >= 2).mean(),
            "Prob. almeno 3x": (port >= 3).mean(),
            "5° percentile": np.quantile(port, 0.05),
            "95° percentile": np.quantile(port, 0.95),
            # Una "fund returner" è una partecipazione che da sola restituisce l'intero portafoglio.
            "Prob. di una fund returner": (m.max(axis=1) >= n).mean(),
        })
    return pd.DataFrame(righe).set_index("Start-up")


def distribuzione_portafoglio(n: int, distr: DistribuzioneRendimenti | None = None, simulazioni: int = 20_000,
                              seme: int = 7) -> np.ndarray:
    distr = distr or DistribuzioneRendimenti()
    rng = np.random.default_rng(seme)
    return distr.campiona((simulazioni, n), rng).mean(axis=1)


def concentrazione(distr: DistribuzioneRendimenti | None = None, n: int = 100_000, seme: int = 11) -> pd.Series:
    """Quota dei proventi totali generata dal miglior 1%, 5%, 10% e 20% degli investimenti."""
    distr = distr or DistribuzioneRendimenti()
    m = np.sort(distr.campiona(n, np.random.default_rng(seme)))[::-1]
    tot = m.sum()
    return pd.Series({f"Miglior {int(q * 100)}%": m[: int(n * q)].sum() / tot for q in (0.01, 0.05, 0.10, 0.20)})


def con_follow_on(
    n: int,
    riserva: float,
    distr: DistribuzioneRendimenti | None = None,
    step_up: float = 2.5,
    prob_segnale_vincenti: float = 0.75,
    prob_segnale_altre: float = 0.25,
    soglia_vincente: float = 3.0,
    simulazioni: int = 20_000,
    seme: int = 7,
) -> pd.Series:
    """Effetto di una riserva per i follow-on.

    Una quota ``riserva`` del capitale è tenuta da parte e reinvestita in parti uguali nelle start-up che
    raccolgono un round successivo. Le migliori (multiplo finale ≥ ``soglia_vincente``) lo raccolgono con
    probabilità più alta: è l'informazione che l'investitore guadagna stando nel capitale. Il follow-on entra a
    un prezzo ``step_up`` volte più alto, quindi rende ``multiplo / step_up``.
    """
    distr = distr or DistribuzioneRendimenti()
    rng = np.random.default_rng(seme)
    m = distr.campiona((simulazioni, n), rng)
    p = np.where(m >= soglia_vincente, prob_segnale_vincenti, prob_segnale_altre)
    scelte = rng.random((simulazioni, n)) < p
    k = scelte.sum(axis=1)
    iniziale = (1 - riserva) * m.mean(axis=1)
    follow = np.where(k > 0, riserva * (scelte * m / step_up).sum(axis=1) / np.maximum(k, 1), riserva)
    port = iniziale + follow
    base = m.mean(axis=1)
    return pd.Series({
        "Multiplo mediano senza riserva": np.median(base), "Multiplo mediano con riserva": np.median(port),
        "Multiplo medio senza riserva": base.mean(), "Multiplo medio con riserva": port.mean(),
        "Prob. perdita senza riserva": (base < 1).mean(), "Prob. perdita con riserva": (port < 1).mean(),
    })
