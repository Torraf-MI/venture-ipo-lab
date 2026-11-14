"""venture-ipo-lab: dalla start-up alla quotazione.

Cap table e round, waterfall con liquidation preference, valutazione con metodo VC e First Chicago, portafogli
angel a legge di potenza, fondi di venture capital (Takahashi-Alexander, J-curve, PME) e analisi empirica del
legame tra private market e IPO.
"""

from .cap_table import CapTable, Classe, Safe
from .dati import carica_ipo, carica_serie, genera_ipo, genera_serie
from .esempi import CASO_DEMO, carica_caso, costruisci_cap_table
from .fondo import ParametriFondo, anno_pareggio, flussi_fondo, indicatori, piano_commitment, scenari_fondo
from .ipo import (
                  chow,
                  confronto_backing,
                  eta_alla_quotazione,
                  ftest_privato,
                  multivariata,
                  prepara,
                  regressione_underpricing,
                  replica_tesi,
                  ricerca_rottura,
                  ttest_backing,
)
from .portafoglio import DistribuzioneRendimenti, con_follow_on, concentrazione, simula_portafogli
from .valutazione import Scenario, first_chicago, metodo_vc, sensitivita_vc
from .waterfall import distribuisci, distribuisci_classi, multipli_exit, soglia_conversione, tabella_exit

__version__ = "0.1.0"

__all__ = [
    "CapTable", "Classe", "Safe", "carica_ipo", "carica_serie", "genera_ipo", "genera_serie", "CASO_DEMO",
    "carica_caso", "costruisci_cap_table", "ParametriFondo", "anno_pareggio", "flussi_fondo", "indicatori",
    "piano_commitment", "scenari_fondo", "confronto_backing", "eta_alla_quotazione", "multivariata", "prepara",
    "regressione_underpricing", "replica_tesi", "ricerca_rottura", "ttest_backing", "chow", "ftest_privato",
    "DistribuzioneRendimenti", "concentrazione", "con_follow_on", "simula_portafogli", "Scenario", "first_chicago",
    "metodo_vc", "sensitivita_vc", "distribuisci", "distribuisci_classi", "multipli_exit", "soglia_conversione",
    "tabella_exit",
]
