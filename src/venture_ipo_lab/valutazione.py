"""Valutazione di una start-up: metodo VC e metodo First Chicago."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EsitoVC:
    valore_exit: float
    multiplo_target: float
    ritenzione: float
    post_money: float
    pre_money: float
    quota_richiesta: float  # quota da chiedere oggi, prima della diluizione futura
    quota_all_exit: float

    def come_serie(self) -> pd.Series:
        return pd.Series({
            "Valore di exit": self.valore_exit, "Multiplo richiesto": self.multiplo_target,
            "Ritenzione dopo i round futuri": self.ritenzione, "Post-money": self.post_money,
            "Pre-money": self.pre_money, "Quota da chiedere oggi": self.quota_richiesta,
            "Quota all'exit": self.quota_all_exit,
        })


def ritenzione(diluizioni_future: list[float]) -> float:
    """Quota che resta all'investitore dopo i round futuri, es. [0.2, 0.15] -> 0.8 × 0.85."""
    return float(np.prod([1 - d for d in diluizioni_future]))


def multiplo_da_tir(tir: float, anni: float) -> float:
    return (1 + tir) ** anni


def metodo_vc(
    valore_exit: float,
    investimento: float,
    multiplo_target: float | None = None,
    tir_target: float | None = None,
    anni: float | None = None,
    diluizioni_future: list[float] | None = None,
) -> EsitoVC:
    """Metodo VC: il post-money oggi è il valore di exit diviso il multiplo richiesto, corretto per la diluizione.

    Il multiplo si indica direttamente o come TIR obiettivo su un orizzonte in anni.
    """
    if multiplo_target is None:
        if tir_target is None or anni is None:
            raise ValueError("Indica multiplo_target oppure tir_target e anni")
        multiplo_target = multiplo_da_tir(tir_target, anni)
    r = ritenzione(diluizioni_future or [])
    post = valore_exit * r / multiplo_target
    if post <= investimento:
        raise ValueError("Il valore di exit non giustifica l'investimento al multiplo richiesto")
    quota = investimento / post
    return EsitoVC(valore_exit, multiplo_target, r, post, post - investimento, quota, quota * r)


@dataclass
class Scenario:
    nome: str
    probabilita: float
    valore_exit: float
    anni: float


def first_chicago(scenari: list[Scenario], tasso: float, investimento: float | None = None,
                  diluizioni_future: list[float] | None = None) -> tuple[pd.DataFrame, float]:
    """Metodo First Chicago: media dei valori attualizzati degli scenari pesata per la probabilità.

    Restituisce la tabella degli scenari e il valore post-money di oggi (già corretto per la diluizione).
    """
    tot = sum(s.probabilita for s in scenari)
    if abs(tot - 1) > 1e-6:
        raise ValueError("Le probabilità degli scenari devono sommare a 1")
    r = ritenzione(diluizioni_future or [])
    righe = []
    for s in scenari:
        vp = s.valore_exit * r / (1 + tasso) ** s.anni
        righe.append({"Scenario": s.nome, "Probabilità": s.probabilita, "Valore di exit": s.valore_exit,
                      "Anni": s.anni, "Valore attuale": vp, "Contributo": s.probabilita * vp})
    df = pd.DataFrame(righe).set_index("Scenario")
    valore = float(df["Contributo"].sum())
    if investimento is not None:
        df.attrs["quota_richiesta"] = investimento / valore
    return df, valore


def sensitivita_vc(valore_exit: float, investimento: float, multipli: list[float],
                   ritenzioni: list[float]) -> pd.DataFrame:
    """Pre-money al variare del multiplo richiesto (righe) e della ritenzione (colonne)."""
    return pd.DataFrame(
        {r: [valore_exit * r / m - investimento for m in multipli] for r in ritenzioni},
        index=pd.Index(multipli, name="Multiplo richiesto"),
    )
