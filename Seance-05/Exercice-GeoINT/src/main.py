
# coding: utf-8

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

DOSSIER = Path(__file__).resolve().parent / "data"


# ============================================================
# FONCTIONS
# ============================================================

def ouvrirUnFichier(nom):
    return pd.read_csv(DOSSIER / nom, low_memory=False)


def conversionLog(liste):
    return [math.log(float(element)) for element in liste]


def ordreDecroissant(liste):
    liste.sort(reverse=True)
    return liste


def ordrePopulation(pop, etat):
    valeurs = []
    for valeur, pays in zip(pop, etat):
        if pd.notna(valeur) and pd.notna(pays):
            valeurs.append([float(valeur), str(pays)])

    ordreDecroissant(valeurs)

    return [[i + 1, ligne[1]] for i, ligne in enumerate(valeurs)]


def classementPays(ordre1, ordre2):
    correspondance = {pays: rang for rang, pays in ordre2}

    classement = [
        [rang, correspondance[pays], pays]
        for rang, pays in ordre1
        if pays in correspondance
    ]

    classement.sort(key=lambda ligne: ligne[0])
    return classement


def colonnesRangs(classement):
    rang1 = []
    rang2 = []

    for ligne in classement:
        rang1.append(ligne[0])
        rang2.append(ligne[1])

    return rang1, rang2


def analyserClassements(ordre1, ordre2, titre):
    classement = classementPays(ordre1, ordre2)
    rang1, rang2 = colonnesRangs(classement)

    if len(rang1) < 2:
        raise ValueError("Pas assez de pays communs")

    spearman = scipy.stats.spearmanr(rang1, rang2)
    kendall = scipy.stats.kendalltau(rang1, rang2)

    print(f"{titre} : {len(rang1)} pays communs")
    print(f"Spearman = {spearman.statistic:.6f}")
    print(f"Kendall = {kendall.statistic:.6f}")

    return spearman.statistic, kendall.statistic


def nombres(colonne):
    return pd.to_numeric(
        colonne, errors="coerce"
    ).tolist()


# ============================================================
# QUESTION 2
# Ouvrir le fichier des îles
# ============================================================

print("\nQuestion 2")

iles = ouvrirUnFichier("island-index.csv")

print(iles.head())


# ============================================================
# QUESTION 3
# Extraire les surfaces et ajouter les continents
# ============================================================

print("\nQuestion 3")

surfaces = pd.to_numeric(
    iles["Surface (km²)"],
    errors="coerce"
).dropna().astype(float).tolist()

surfaces.extend([
    85545323.0,
    37856841.0,
    7768030.0,
    7605049.0
])

surfaces = [valeur for valeur in surfaces if valeur > 0]

print("Nombre de surfaces :", len(surfaces))


# ============================================================
# QUESTION 4
# Classer les surfaces par ordre décroissant
# ============================================================

print("\nQuestion 4")

ordreDecroissant(surfaces)

print("Dix plus grandes surfaces :", surfaces[:10])


# ============================================================
# QUESTION 5
# Créer la loi rang-taille
# ============================================================

print("\nQuestion 5")

rangs = list(range(1, len(surfaces) + 1))

plt.figure(figsize=(9, 5))
plt.plot(rangs, surfaces)

plt.xlabel("Rang")
plt.ylabel("Surface (km²)")
plt.title("Loi rang-taille des îles et continents")

plt.grid(True)
plt.tight_layout()

plt.savefig("question_5_rang_taille.png", dpi=200)
plt.close()

print("Graphique enregistré")


# ============================================================
# QUESTION 6
# Convertir les axes en logarithmes
# ============================================================

print("\nQuestion 6")

plt.figure(figsize=(9, 5))

plt.plot(
    conversionLog(rangs),
    conversionLog(surfaces)
)

plt.xlabel("Logarithme du rang")
plt.ylabel("Logarithme de la surface (km²)")
plt.title("Loi rang-taille en logarithmes")

plt.grid(True)
plt.tight_layout()

plt.savefig("question_6_rang_taille_log.png", dpi=200)
plt.close()

print("Graphique logarithmique enregistré")


# ============================================================
# QUESTION 7
# Réponse argumentée
# ============================================================

print("\nQuestion 7")

# Oui, il est possible de réaliser un test statistique
# sur les rangs.
#
# Les coefficients de Spearman et de Kendall permettent
# de comparer deux classements portant sur les mêmes objets.
#
# Spearman mesure la corrélation entre les rangs.
# Kendall mesure leur concordance.
#
# Il faut toutefois disposer de deux classements
# pertinents ou formuler une hypothèse à tester.


# ============================================================
# QUESTION 8
# Le second fichier CSV est placé dans src/data
# ============================================================


# ============================================================
# QUESTION 9
# Ouvrir le fichier des États du monde
# ============================================================

print("\nQuestion 9")

monde = ouvrirUnFichier(
    "Le-Monde-HS-Etats-du-monde-2007-2026.csv"
)

print(monde.head())


# ============================================================
# QUESTION 10
# Isoler les colonnes demandées
# ============================================================

print("\nQuestion 10")

etats = monde["État"].tolist()

pop2007 = nombres(monde["Pop 2007"])
pop2026 = nombres(monde["Pop 2026"])

densite2007 = nombres(monde["Densité 2007"])
densite2026 = nombres(monde["Densité 2026"])

print("Nombre d'États :", len(etats))


# ============================================================
# QUESTION 11
# Classer les populations et les densités
# ============================================================

print("\nQuestion 11")

ordre_pop2007 = ordrePopulation(pop2007, etats)
ordre_pop2026 = ordrePopulation(pop2026, etats)

ordre_densite2007 = ordrePopulation(densite2007, etats)
ordre_densite2026 = ordrePopulation(densite2026, etats)

print("Population 2007 :", ordre_pop2007[:10])
print("Population 2026 :", ordre_pop2026[:10])

print("Densité 2007 :", ordre_densite2007[:10])
print("Densité 2026 :", ordre_densite2026[:10])


# ============================================================
# QUESTION 12
# Comparer les classements
# ============================================================

print("\nQuestion 12")

classement_population = classementPays(
    ordre_pop2007,
    ordre_pop2026
)

classement_densite = classementPays(
    ordre_densite2007,
    ordre_densite2026
)

print("Population :", classement_population[:10])
print("Densité :", classement_densite[:10])


# ============================================================
# QUESTION 13
# Isoler les deux colonnes de rangs
# ============================================================

print("\nQuestion 13")

rang_pop2007, rang_pop2026 = colonnesRangs(
    classement_population
)

rang_densite2007, rang_densite2026 = colonnesRangs(
    classement_densite
)

print("Rangs population 2007 :", rang_pop2007[:10])
print("Rangs population 2026 :", rang_pop2026[:10])

print("Rangs densité 2007 :", rang_densite2007[:10])
print("Rangs densité 2026 :", rang_densite2026[:10])


# ============================================================
# QUESTION 14
# Spearman et Kendall
# ============================================================

print("\nQuestion 14")

analyserClassements(
    ordre_pop2007,
    ordre_pop2026,
    "Population 2007-2026"
)

analyserClassements(
    ordre_densite2007,
    ordre_densite2026,
    "Densité 2007-2026"
)


# ============================================================
# BONUS 1
# Comparer la surface des îles et leur trait de côte
# ============================================================

print("\nBonus 1")

iles_bonus = iles[
    ["Surface (km²)", "Trait de côte (km)"]
].apply(
    pd.to_numeric,
    errors="coerce"
).dropna()

iles_bonus = iles_bonus[
    (iles_bonus["Surface (km²)"] > 0)
    & (iles_bonus["Trait de côte (km)"] > 0)
]

spearman_iles = scipy.stats.spearmanr(
    iles_bonus["Surface (km²)"],
    iles_bonus["Trait de côte (km)"]
)

kendall_iles = scipy.stats.kendalltau(
    iles_bonus["Surface (km²)"],
    iles_bonus["Trait de côte (km)"]
)

print("Spearman :", spearman_iles.statistic)
print("Kendall :", kendall_iles.statistic)


# ============================================================
# BONUS 2
# Comparer tous les classements annuels de 2007 à 2026
# ============================================================

print("\nBonus 2")

classements_annuels = {}

for annee in range(2007, 2027):
    population = nombres(monde[f"Pop {annee}"])

    classements_annuels[annee] = ordrePopulation(
        population,
        etats
    )

resultats = []

for annee1 in range(2007, 2027):
    for annee2 in range(annee1 + 1, 2027):

        classement = classementPays(
            classements_annuels[annee1],
            classements_annuels[annee2]
        )

        rang1, rang2 = colonnesRangs(classement)

        if len(rang1) >= 2:
            spearman = scipy.stats.spearmanr(
                rang1, rang2
            ).statistic

            kendall = scipy.stats.kendalltau(
                rang1, rang2
            ).statistic

            resultats.append([
                annee1,
                annee2,
                spearman,
                kendall
            ])

resultats = pd.DataFrame(
    resultats,
    columns=[
        "Année 1",
        "Année 2",
        "Spearman",
        "Kendall"
    ]
)

resultats.to_csv(
    "bonus_comparaison_2007_2026.csv",
    index=False
)

print("Comparaisons enregistrées :", len(resultats))
print("Fichier bonus enregistré")

