"""Riga di comando: report su start-up, portafoglio angel, fondo di venture capital e IPO.

Esempi::

    venture-ipo-lab --demo
    venture-ipo-lab --caso esempi/startup.json --patrimonio 2000000 --allocazione 0.05
    venture-ipo-lab --demo --serie data/serie_esempio.csv --ipo data/ipo_esempio.csv --offline
"""

from __future__ import annotations

import argparse
import sys

from . import dati, esempi
from .report import _eur, _num, _pct, analizza, build_report, salva_grafici


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="venture-ipo-lab", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="Usa il caso dimostrativo della start-up")
    src.add_argument("--caso", help="File JSON con fondatori, SAFE, round, scenari di exit (vedi esempi/)")
    p.add_argument("--serie", help="CSV con proprie serie annuali per regione (es. export Bloomberg)")
    p.add_argument("--ipo", help="CSV con il campione di singole IPO (underpricing e rendimenti)")
    p.add_argument("--patrimonio", type=float, default=5_000_000, help="Patrimonio per il piano di commitment")
    p.add_argument("--allocazione", type=float, default=0.10, help="Quota obiettivo in venture capital, es. 0.10")
    p.add_argument("--simulazioni", type=int, default=20_000, help="Simulazioni per i portafogli angel")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    caso = esempi.CASO_DEMO if args.demo else esempi.carica_caso(args.caso)
    serie = dati.carica_serie(args.serie) if args.serie else None
    campione = dati.carica_ipo(args.ipo) if args.ipo else None
    a = analizza(caso, serie, campione, args.patrimonio, args.allocazione, args.simulazioni)
    out = build_report(a, args.output, offline=args.offline)

    q = a.cap_table.quote_per_titolare()
    print(f"Caso: {caso['nome']}")
    print(f"  Quota dei fondatori: {_pct(sum(q[f] for f in caso['fondatori']), 0)}"
          f" · valore First Chicago: {_eur(a.valore_fc)}")
    p = a.portafogli
    print(f"  Probabilità di perdere capitale: {_pct(p.loc[10, 'Prob. di perdere capitale'], 0)} con 10 start-up,"
          f" {_pct(p.loc[50, 'Prob. di perdere capitale'], 0)} con 50")
    print(f"  Fondo: TVPI {_num(a.indicatori_fondo['TVPI'])}x, TIR {_pct(a.indicatori_fondo['TIR'])},"
          f" commitment iniziale {_eur(a.commitment)}")
    e = a.empirico
    print(f"  IPO USA (dati pubblici): età mediana {_num(e.eta['media prima'], 1)} → {_num(e.eta['media dopo'], 1)}"
          f" anni; quota VC/PE non significativa a parità di mercato (p = {_num(e.f_mercati_privati['p-value'], 2)});"
          f" rottura nel {e.rottura['anno']} (sup-F {_num(e.rottura['sup-F'], 1)})")
    if a.pannello:
        for reg, t in a.test_f.items():
            print(f"  Dati propri, {reg}: effetto del capitale privato {_num(t['Effetto prima del 2008'], 2, True)}"
                  f" prima del 2008, {_num(t['Effetto dopo il 2008'], 2, True)} dopo")
    print(f"Report: {out}")
    if args.grafici:
        for f in salva_grafici(a, args.grafici):
            print(f"Grafico: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
