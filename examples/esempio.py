"""Uso della libreria: una start-up dal SAFE alla Serie A, il suo waterfall e i test sulla relazione IPO-privato."""

from venture_ipo_lab import (
    CapTable,
    chow,
    distribuisci,
    ftest_privato,
    genera_serie,
    metodo_vc,
    prepara,
    simula_portafogli,
)

# 1. Cap table: due fondatori, un SAFE e un round seed con ampliamento del pool
ct = CapTable.fondazione({"Fondatrice": 0.6, "Fondatore": 0.4})
ct.safe("Business angel", 200_000, cap=2_500_000, sconto=0.2)
ct.round("Seed", 3_500_000, {"Fondo seed": 1_000_000}, pool_post=0.10)
ct.round("Serie A", 12_000_000, {"Fondo VC": 4_000_000}, pool_post=0.12, partecipante=True, cap_partecipazione=3)
print(ct.diluizione().round(3), "\n")

# 2. Chi riceve cosa con una vendita a 20 e a 100 milioni
for valore in (20e6, 100e6):
    print(f"Exit a {valore / 1e6:.0f} milioni:\n{distribuisci(ct, valore).round(0)}\n")

# 3. Metodo VC: quanto chiedere per 4 milioni con un obiettivo del 40% annuo su 6 anni
print(metodo_vc(150e6, 4e6, tir_target=0.40, anni=6, diluizioni_future=[0.2, 0.15]).come_serie(), "\n")

# 4. Portafoglio angel: rischio di perdita al variare del numero di start-up
print(simula_portafogli([5, 20, 50], simulazioni=10_000)[["Multiplo mediano", "Prob. di perdere capitale"]], "\n")

# 5. IPO e private market (dati sintetici dimostrativi; con dati veri: carica_serie("file.csv"))
df = prepara(genera_serie(), "Europa occidentale")
print(ftest_privato(df).round(3))
print(chow(df).round(3))
