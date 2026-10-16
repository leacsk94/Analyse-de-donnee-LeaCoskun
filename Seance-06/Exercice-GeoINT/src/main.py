
# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


# ============================================================
# FONCTIONS
# ============================================================

DOSSIER = Path(__file__).resolve().parent
DOSSIER_DATA = DOSSIER / "data"
DOSSIER_RESULTATS = DOSSIER / "resultats"

DOSSIER_RESULTATS.mkdir(exist_ok=True)


def ouvrirUnFichier(nom):
    return pd.read_csv(DOSSIER_DATA / nom)


def nettoyerDonnees(tableau, colonne_x, colonne_y):
    # On conserve uniquement les deux colonnes demandées.
    donnees = tableau[[colonne_x, colonne_y]].copy()

    # On transforme les valeurs en nombres.
    # Les valeurs non numériques deviennent NaN.
    donnees[colonne_x] = pd.to_numeric(
        donnees[colonne_x], errors="coerce"
    )
    donnees[colonne_y] = pd.to_numeric(
        donnees[colonne_y], errors="coerce"
    )

    # On remplace les valeurs infinies par NaN.
    donnees = donnees.replace([np.inf, -np.inf], np.nan)

    # On retire les lignes où une valeur manque.
    donnees = donnees.dropna()

    return donnees


def calculerStatistiques(donnees, colonne_x, colonne_y):
    x = donnees[colonne_x].to_numpy()
    y = donnees[colonne_y].to_numpy()

    regression = scipy.stats.linregress(x, y)
    correlation = scipy.stats.pearsonr(x, y)

    return regression, correlation


def creerGraphique(donnees, regression, annee, fichier):
    x = donnees[f"Utilisation_d_energie_{annee}"].to_numpy()
    y = donnees[f"PIB_{annee}"].to_numpy()

    # Valeurs utilisées pour tracer la droite de régression.
    x_droite = np.linspace(x.min(), x.max(), 100)
    y_droite = regression.slope * x_droite + regression.intercept

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x, y,
        alpha=0.6,
        label="Territoires"
    )

    plt.plot(
        x_droite,
        y_droite,
        color="red",
        label="Droite de régression"
    )

    plt.xlabel("Consommation énergétique (kg équivalent pétrole)")
    plt.ylabel("PIB (dollars courants)")
    plt.title(f"PIB et consommation énergétique en {annee}")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fichier, dpi=200)
    plt.close()


# ============================================================
# QUESTION 1
# Sélectionner les colonnes de 2022
# ============================================================

print("\nQuestion 1")

monde = ouvrirUnFichier("pib-vs-energie.csv")

print("Nombre de territoires :", len(monde))

pib2022 = monde["PIB_2022"]
energie2022 = monde["Utilisation_d_energie_2022"]

print("PIB 2022 :")
print(pib2022.head())

print("Énergie 2022 :")
print(energie2022.head())


# ============================================================
# QUESTION 2
# Supprimer les données manquantes ou partielles
# ============================================================

print("\nQuestion 2")

donnees2022 = nettoyerDonnees(
    monde,
    "Utilisation_d_energie_2022",
    "PIB_2022"
)

print("Nombre de couples complets :", len(donnees2022))
print("Nombre de lignes exclues :", len(monde) - len(donnees2022))

print(donnees2022.head())


# ============================================================
# QUESTION 3
# Régression linéaire simple
# ============================================================

print("\nQuestion 3")

# x est la variable explicative : consommation énergétique.
# y est la variable à expliquer : PIB.

x = donnees2022["Utilisation_d_energie_2022"]
y = donnees2022["PIB_2022"]

regression2022 = scipy.stats.linregress(x, y)

print("Pente :", regression2022.slope)
print("Ordonnée à l'origine :", regression2022.intercept)
print("Coefficient R² :", regression2022.rvalue ** 2)

print(
    "Équation : PIB =",
    regression2022.slope,
    "* énergie +",
    regression2022.intercept
)


# ============================================================
# QUESTION 4
# Corrélation simple
# ============================================================

print("\nQuestion 4")

correlation2022 = scipy.stats.pearsonr(x, y)

print("Coefficient de Pearson :", correlation2022.statistic)
print("p-value :", correlation2022.pvalue)

# Le coefficient varie entre -1 et 1.
# Une valeur proche de 1 indique une association positive forte.
# Une valeur proche de 0 indique une faible association linéaire.
# Une valeur proche de -1 indique une association négative forte.


# ============================================================
# QUESTION 5
# Graphique de synthèse avec droite de régression
# ============================================================

print("\nQuestion 5")

creerGraphique(
    donnees2022,
    regression2022,
    2022,
    DOSSIER_RESULTATS / "question_5_regression_2022.png"
)

print("Graphique enregistré dans le dossier resultats")


# ============================================================
# QUESTION 6
# Commentaire des résultats
# ============================================================

print("\nQuestion 6")

r = correlation2022.statistic

if r > 0:
    print("La relation observée est positive.")
elif r < 0:
    print("La relation observée est négative.")
else:
    print("Aucune corrélation linéaire observée.")

print("Coefficient de corrélation :", round(r, 4))
print("Coefficient de détermination R² :", round(r ** 2, 4))

# Le commentaire détaillé sera rédigé dans le rapport.
# Il faudra interpréter la corrélation, la régression
# et la dispersion des points sur le graphique.
# Une corrélation ne prouve pas une relation de causalité.


# ============================================================
# BONUS
# Généraliser les calculs aux années 1962 à 2022
# ============================================================

print("\nBonus")

resultats = []

DOSSIER_GRAPHIQUES = DOSSIER_RESULTATS / "graphiques_bonus"
DOSSIER_GRAPHIQUES.mkdir(exist_ok=True)

for annee in range(1962, 2023):

    colonne_x = f"Utilisation_d_energie_{annee}"
    colonne_y = f"PIB_{annee}"

    if colonne_x not in monde.columns or colonne_y not in monde.columns:
        print(f"{annee} : colonnes absentes")
        continue

    donnees = nettoyerDonnees(
        monde,
        colonne_x,
        colonne_y
    )

    if len(donnees) < 3:
        print(f"{annee} : données insuffisantes")
        continue

    # Une régression nécessite des valeurs x différentes.
    if donnees[colonne_x].nunique() < 2:
        print(f"{annee} : énergie constante")
        continue

    if donnees[colonne_y].nunique() < 2:
        print(f"{annee} : PIB constant")
        continue

    regression, correlation = calculerStatistiques(
        donnees,
        colonne_x,
        colonne_y
    )

    resultats.append({
        "Annee": annee,
        "Nombre_territoires": len(donnees),
        "Pente": regression.slope,
        "Ordonnee_origine": regression.intercept,
        "Pearson": correlation.statistic,
        "R2": regression.rvalue ** 2,
        "P_value": correlation.pvalue
    })

    creerGraphique(
        donnees,
        regression,
        annee,
        DOSSIER_GRAPHIQUES / f"regression_{annee}.png"
    )

    print(
        f"{annee} : {len(donnees)} territoires, "
        f"Pearson = {correlation.statistic:.4f}"
    )


# ============================================================
# ENREGISTREMENT DES RÉSULTATS DU BONUS
# ============================================================

tableau_resultats = pd.DataFrame(resultats)

tableau_resultats.to_csv(
    DOSSIER_RESULTATS / "bonus_regressions_1962_2022.csv",
    index=False
)

print("\nBonus terminé")
print("Nombre d'années analysées :", len(tableau_resultats))
print("Résultats enregistrés dans le dossier resultats")

