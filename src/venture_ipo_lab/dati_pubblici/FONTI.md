# Fonti dei dati

Tutte le serie sono annuali, trascritte da pubblicazioni pubbliche e gratuite. Nessun dato proviene da terminali a
pagamento; i dati Bloomberg usati nella tesi del 2019 non sono ripubblicati (il pacchetto offre un caricatore per
chi ha un proprio export, vedi `venture_ipo_lab.empirico.carica_export`).

## `ipo_usa.csv` (1980-2025)

*Initial Public Offerings: Updated Statistics*, Warrington College of Business, University of Florida, versione del 5 novembre 2026
(https://site.warrington.ufl.edu/ritter/ipo-data/). Campione: IPO con prezzo di almeno 5 dollari, esclusi ADR,
unit, SPAC, REIT, fondi chiusi, banche e casse di risparmio, IPO non quotate su NYSE, Nasdaq o NYSE MKT.

| Colonna | Tabella | Contenuto |
|---|---|---|
| `ipo` | 1 | numero di IPO |
| `rendimento_primo_giorno_medio`, `_mediano` | 1 | rendimento del primo giorno di quotazione, % (equipesato) |
| `denaro_lasciato_mld_usd` | 1 | *money left on the table*: (prezzo di chiusura del primo giorno − prezzo di offerta) × azioni offerte, miliardi di dollari |
| `raccolta_mld_usd` | 1 | raccolta lorda, miliardi di dollari nominali |
| `ipo_sponsor_finanziario`, `ipo_vc`, `ipo_growth`, `ipo_buyout` | 4c | IPO sostenute da un investitore finanziario e per tipo: venture capital, growth capital, buyout (private equity) |
| `eta_mediana` | 4 | età mediana (anni dalla fondazione) delle società alla IPO |
| `eta_mediana_vc_tech` | 4i | età mediana delle IPO tecnologiche sostenute da VC |

Le tabelle della pubblicazione non coincidono sempre tra loro (per esempio il 2017 conta 107 IPO nella tabella 1 e 106
nella 4c); qui ogni colonna riporta la propria tabella.

## `macro.csv` (1980-2025)

| Colonna | Fonte |
|---|---|
| `sp500_rendimento_totale` | *Historical Returns on Stocks, Bonds and Bills*, NYU Stern School of Business (pagina datafile `histretSP`), aggiornamento del 5 gennaio 2026 (rendimento dell'S&P 500 con dividendi, %) |
| `treasury_10a`, `fed_funds` | FRED, Federal Reserve Bank of St. Louis: GS10 e FEDFUNDS, media annua, % |
| `vix` | FRED, VIXCLS, media annua (dal 1990) |
| `cape_gennaio` | multpl.com, CAPE (P/E corretto per il ciclo), valore di gennaio |
| `bund_10a` | FRED, IRLTLT01DEM156N (OCSE), rendimento dei titoli di Stato tedeschi a lungo termine, media annua, % |
| `dax_variazione` | Wikipedia, voce *DAX*, variazione annua dell'indice, % |

## `mercati_privati.csv` (2014-2025)

Rapporti pubblici di PitchBook, valori stampati sulle etichette dei grafici (i conteggi includono stime di
PitchBook per gli anni recenti).

| Colonne | Rapporto |
|---|---|
| `usa_vc_*` | PitchBook-NVCA *Venture Monitor* Q4 2025, grafico *VC deal activity* (2015-2025) |
| `usa_pe_*` | PitchBook *2025 Annual US PE Breakdown*, grafico *PE deal activity* (2015-2025) |
| `eur_vc_*` | PitchBook *2025 Annual European Venture Report*: operazioni, exit (il 2025 comprende la quotazione di Klarna) e raccolta dei fondi (2015-2025) |
| `eur_pe_*` | PitchBook *2024 Annual European PE Breakdown*: operazioni, exit, IPO sostenute da PE e raccolta (2014-2024) |

## Uso e licenze

Le serie sono valori numerici (fatti) trascritti dalle pubblicazioni indicate, citate con la fonte, a scopo di
studio e non commerciale. I diritti sulle pubblicazioni restano ai rispettivi editori; prima di un uso commerciale
vanno verificati i termini di ciascuna fonte:

- statistiche IPO della University of Florida e dati della NYU Stern School of Business: pagine accademiche
  pubbliche e gratuite, da citare;
- FRED: si applicano i termini d'uso di FRED; il VIX è un indice di Cboe Global Markets, riportato tramite FRED;
- multpl.com: valori del CAPE pubblici, da citare;
- Wikipedia: licenza CC BY-SA 4.0; qui si riportano solo le variazioni annue del DAX (fatti numerici);
- PitchBook: pochi valori aggregati annuali letti dai grafici di rapporti gratuiti, citati con il titolo del
  rapporto; i rapporti restano di proprietà di PitchBook Data, Inc. e non sono ridistribuiti.
