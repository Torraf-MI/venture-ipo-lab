"""Cap table di una start-up: fondatori, SAFE, round priced, option pool e diluizione round per round.

Convenzioni:

* le quote sono calcolate sul capitale *fully diluted* (azioni emesse più l'intero option pool);
* il prezzo di un round è ``pre-money / azioni fully diluted prima del round``, dove le azioni prima del round
  includono la conversione dei SAFE e l'eventuale ampliamento del pool (il cosiddetto *option pool shuffle*:
  il pool richiesto dagli investitori lo pagano i soci esistenti, perché entra nel pre-money);
* i SAFE sono *post-money SAFE* (lo standard di Y Combinator dal 2018): al cap, l'investitore ottiene la quota
  ``importo / cap`` del capitale prima del round; con lo sconto paga il prezzo del round ridotto. Vale il prezzo
  più basso tra i due.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

POOL = "Option pool"
COMUNI = "Ordinarie"


@dataclass
class Classe:
    """Una classe di azioni con i suoi diritti economici all'exit."""

    nome: str
    prezzo: float = 0.0  # prezzo per azione pagato dagli investitori
    multiplo_pref: float = 0.0  # liquidation preference come multiplo dell'investito (0 = ordinarie)
    partecipante: bool = False  # dopo la preference partecipa anche al residuo
    cap_partecipazione: float | None = None  # tetto al rendimento totale dei partecipanti (es. 3 = 3x)
    seniority: int = 0  # più alto = pagato prima
    investito: float = 0.0  # capitale versato dalla classe

    @property
    def privilegiata(self) -> bool:
        return self.multiplo_pref > 0


@dataclass
class Safe:
    investitore: str
    importo: float
    cap: float | None = None  # valutazione post-money massima
    sconto: float = 0.0  # sconto sul prezzo del round, es. 0.2


@dataclass
class CapTable:
    """Libro soci con la storia di ogni passaggio."""

    posizioni: dict[tuple[str, str], float] = field(default_factory=dict)  # (titolare, classe) -> azioni
    classi: dict[str, Classe] = field(default_factory=dict)
    safe_in_attesa: list[Safe] = field(default_factory=list)
    storia: list[tuple[str, pd.Series]] = field(default_factory=list)
    round_info: list[dict] = field(default_factory=list)
    versato: dict[tuple[str, str], float] = field(default_factory=dict)  # (titolare, classe) -> capitale versato

    # ------------------------------------------------------------------ #
    # Costruzione
    # ------------------------------------------------------------------ #
    @classmethod
    def fondazione(cls, fondatori: dict[str, float], azioni: float = 10_000_000, pool: float = 0.0) -> CapTable:
        """Crea la società: ``fondatori`` con le quote iniziali (sommano a 1), ``pool`` la quota per le opzioni."""
        tot = sum(fondatori.values())
        if abs(tot - 1) > 1e-9:
            raise ValueError("Le quote dei fondatori devono sommare a 1")
        ct = cls()
        ct.classi[COMUNI] = Classe(COMUNI)
        quota_fondatori = 1 - pool
        for nome, q in fondatori.items():
            ct.posizioni[(nome, COMUNI)] = azioni * quota_fondatori * q
        if pool > 0:
            ct.posizioni[(POOL, COMUNI)] = azioni * pool
        ct._registra("Costituzione")
        return ct

    def copia(self) -> CapTable:
        return deepcopy(self)

    @property
    def azioni_totali(self) -> float:
        return sum(self.posizioni.values())

    def azioni_classe(self, classe: str) -> float:
        return sum(a for (_, c), a in self.posizioni.items() if c == classe)

    def _aggiungi(self, titolare: str, classe: str, azioni: float) -> None:
        self.posizioni[(titolare, classe)] = self.posizioni.get((titolare, classe), 0.0) + azioni

    def _registra(self, passaggio: str) -> None:
        self.storia.append((passaggio, self.quote_per_titolare()))

    def safe(self, investitore: str, importo: float, cap: float | None = None, sconto: float = 0.0) -> CapTable:
        """Registra un SAFE: si converte al primo round priced successivo."""
        if cap is None and sconto <= 0:
            raise ValueError("Un SAFE deve avere un cap, uno sconto o entrambi")
        self.safe_in_attesa.append(Safe(investitore, importo, cap, sconto))
        return self

    def round(
        self,
        nome: str,
        pre_money: float,
        investitori: dict[str, float],
        pool_post: float | None = None,
        multiplo_pref: float = 1.0,
        partecipante: bool = False,
        cap_partecipazione: float | None = None,
        seniority: int | None = None,
    ) -> CapTable:
        """Round priced: ``investitori`` mappa ogni investitore all'importo versato.

        ``pool_post`` è la quota di option pool (non assegnato più assegnato) richiesta *dopo* il round: le azioni
        mancanti vengono create prima del round, quindi diluiscono solo i soci esistenti.
        ``seniority`` di default rende ogni nuovo round più senior dei precedenti (struttura *stacked*).
        """
        investimento = sum(investitori.values())
        s_pre = self.azioni_totali
        pool_attuale = self.azioni_classe_titolare(POOL)
        if seniority is None:
            seniority = 1 + max((c.seniority for c in self.classi.values()), default=0)

        # Punto fisso: prezzo, azioni dei SAFE e ampliamento del pool dipendono l'uno dall'altro.
        safe_azioni = [0.0] * len(self.safe_in_attesa)
        nuovo_pool = 0.0
        prezzo = pre_money / s_pre
        for _ in range(200):
            fd_pre = s_pre + sum(safe_azioni) + nuovo_pool
            prezzo = pre_money / fd_pre
            post = fd_pre + investimento / prezzo
            if pool_post is not None:
                nuovo_pool = max(0.0, pool_post * post - pool_attuale)
            capitale_safe = s_pre + sum(safe_azioni)  # capitale su cui si misura la quota al cap
            nuove = []
            for s in self.safe_in_attesa:
                prezzi = []
                if s.cap is not None:
                    prezzi.append(s.cap / capitale_safe)
                if s.sconto > 0:
                    prezzi.append(prezzo * (1 - s.sconto))
                nuove.append(s.importo / min(prezzi))
            if max((abs(a - b) for a, b in zip(nuove, safe_azioni)), default=0) < 1e-6 and abs(
                pre_money / (s_pre + sum(nuove) + nuovo_pool) - prezzo) < 1e-12:
                safe_azioni = nuove
                break
            safe_azioni = nuove
        fd_pre = s_pre + sum(safe_azioni) + nuovo_pool
        prezzo = pre_money / fd_pre

        classe = Classe(nome, prezzo, multiplo_pref, partecipante, cap_partecipazione, seniority)
        self.classi[nome] = classe
        if nuovo_pool > 0:
            self._aggiungi(POOL, COMUNI, nuovo_pool)
        for s, az in zip(self.safe_in_attesa, safe_azioni):
            self._aggiungi(s.investitore, nome, az)
            self.versato[(s.investitore, nome)] = self.versato.get((s.investitore, nome), 0.0) + s.importo
            classe.investito += s.importo
        for inv, importo in investitori.items():
            self._aggiungi(inv, nome, importo / prezzo)
            self.versato[(inv, nome)] = self.versato.get((inv, nome), 0.0) + importo
            classe.investito += importo
        convertiti = sum(s.importo for s in self.safe_in_attesa)
        self.safe_in_attesa = []
        post_money = prezzo * self.azioni_totali
        self.round_info.append({
            "Round": nome, "Pre-money": pre_money, "Investimento": investimento, "SAFE convertiti": convertiti,
            "Post-money": post_money, "Prezzo per azione": prezzo, "Azioni nuove pool": nuovo_pool,
            "Pre-money effettivo": pre_money - nuovo_pool * prezzo,
        })
        self._registra(nome)
        return self

    def azioni_classe_titolare(self, titolare: str) -> float:
        return sum(a for (t, _), a in self.posizioni.items() if t == titolare)

    # ------------------------------------------------------------------ #
    # Viste
    # ------------------------------------------------------------------ #
    def quote_per_titolare(self) -> pd.Series:
        tot = self.azioni_totali
        s = pd.Series({t: 0.0 for t, _ in self.posizioni})
        for (t, _), a in self.posizioni.items():
            s[t] += a / tot
        return s

    def tabella(self) -> pd.DataFrame:
        """Libro soci: una riga per titolare e classe."""
        tot = self.azioni_totali
        righe = [{"Titolare": t, "Classe": c, "Azioni": a, "Quota": a / tot,
                  "Investito": self.versato.get((t, c), 0.0)}
                 for (t, c), a in self.posizioni.items()]
        return pd.DataFrame(righe)

    def diluizione(self) -> pd.DataFrame:
        """Quota fully diluted di ogni titolare dopo ogni passaggio (righe = passaggi)."""
        df = pd.DataFrame({p: q for p, q in self.storia}).T.fillna(0.0)
        return df

    def rounds(self) -> pd.DataFrame:
        return pd.DataFrame(self.round_info).set_index("Round") if self.round_info else pd.DataFrame()
