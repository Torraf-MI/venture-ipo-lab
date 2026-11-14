"""Caso dimostrativo e lettura dei casi da file JSON.

Il caso dimostrativo è inventato: una start-up software con due fondatori, un business angel che entra con un
SAFE, un round seed, una Serie A e una Serie B. I nomi sono generici e non si riferiscono a società reali.
"""

from __future__ import annotations

import json
from pathlib import Path

from .cap_table import CapTable
from .valutazione import Scenario

CASO_DEMO: dict = {
    "nome": "Start-up dimostrativa (software B2B)",
    "fondatori": {"Fondatrice A": 0.55, "Fondatore B": 0.45},
    "pool_iniziale": 0.0,
    "investitore": "Business angel",
    "passaggi": [
        {"tipo": "safe", "investitore": "Business angel", "importo": 250_000, "cap": 3_000_000},
        {"tipo": "round", "nome": "Seed", "pre_money": 4_000_000, "investitori": {"Fondo seed": 1_200_000},
         "pool_post": 0.10},
        {"tipo": "round", "nome": "Serie A", "pre_money": 16_000_000,
         "investitori": {"Fondo VC A": 5_000_000, "Fondo seed": 500_000}, "pool_post": 0.12},
        {"tipo": "round", "nome": "Serie B", "pre_money": 55_000_000, "investitori": {"Fondo growth": 18_000_000},
         "pool_post": 0.12, "multiplo_pref": 1.0, "partecipante": True, "cap_partecipazione": 3.0},
    ],
    "exit": {
        "tasso": 0.35,
        "scenari": [
            {"nome": "Fallimento", "probabilita": 0.35, "valore_exit": 0, "anni": 4},
            {"nome": "Vendita in perdita", "probabilita": 0.30, "valore_exit": 40_000_000, "anni": 5},
            {"nome": "Acquisizione", "probabilita": 0.25, "valore_exit": 220_000_000, "anni": 6},
            {"nome": "Quotazione (IPO)", "probabilita": 0.10, "valore_exit": 900_000_000, "anni": 7},
        ],
    },
    "valutazione_vc": {"valore_exit": 220_000_000, "investimento": 5_500_000, "tir_target": 0.40, "anni": 6,
                       "diluizioni_future": [0.25, 0.15]},
}


def costruisci_cap_table(caso: dict) -> CapTable:
    ct = CapTable.fondazione(caso["fondatori"], pool=caso.get("pool_iniziale", 0.0))
    for p in caso["passaggi"]:
        if p["tipo"] == "safe":
            ct.safe(p["investitore"], p["importo"], p.get("cap"), p.get("sconto", 0.0))
        elif p["tipo"] == "round":
            ct.round(p["nome"], p["pre_money"], p["investitori"], p.get("pool_post"), p.get("multiplo_pref", 1.0),
                     p.get("partecipante", False), p.get("cap_partecipazione"), p.get("seniority"))
        else:
            raise ValueError(f"Tipo di passaggio sconosciuto: {p['tipo']!r}")
    return ct


def scenari(caso: dict) -> list[Scenario]:
    return [Scenario(s["nome"], s["probabilita"], s["valore_exit"], s["anni"]) for s in caso["exit"]["scenari"]]


def carica_caso(percorso: str | Path) -> dict:
    with open(percorso, encoding="utf-8") as f:
        caso = json.load(f)
    for chiave in ("fondatori", "passaggi", "exit"):
        if chiave not in caso:
            raise ValueError(f"Nel caso manca la sezione {chiave!r}")
    return caso
