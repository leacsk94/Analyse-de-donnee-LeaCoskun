
# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats


DOSSIER = Path(__file__).resolve().parent
DATA = DOSSIER / "data"
RESULTATS = DOSSIER / "resultats"

RESULTATS.mkdir(exist_ok=True)


def ouvrirUnFichier(nom):
    return pd.read_csv(DATA / nom)


def enregistrerTableau(tableau, nom):
    tableau.to_csv(
        RESULTATS / nom,
        index=True,
        encoding="utf-8-sig"
    )


def calculerRegression(tableau, cible, variables, nom):
    donnees = tableau[[cible] + variables].copy()
    donnees = donnees.apply(pd.to_numeric, errors="coerce")
    donnees = donnees.replace([np.inf, -np.inf], np.nan)
    donnees = donnees.dropna()

    y = donnees[cible]
    X = donnees[variables]

    # Régression avec Statsmodels
    X_constante = sm.add_constant(X, has_constant="add")
    modele = sm.OLS(y, X_constante).fit()

    print("\nNombre d'observations :", len(donnees))
    print("\nCoefficients :")
    print(modele.params)

    print("\nR2 :", modele.rsquared)
    print("R2 ajusté :", modele.rsquared_adj)

    print("\np-values :")
    print(modele.pvalues)

    print("\nRésumé de la régression :")
    print(modele.summary())

    # Enregistrement des coefficients
    coefficients = pd.DataFrame({
        "Coefficient": modele.params,
        "p-value": modele.pvalues
    })

    enregistrerTableau(
        coefficients,
        nom + "_coefficients.csv"
    )

    # Enregistrement du résumé
    with open(
        RESULTATS / (nom + "_resume.txt"),
        "w",
        encoding="utf-8"
    ) as fichier:
        fichier.write(modele.summary().as_text())

    # Comparaison entre valeurs observées et estimées
    predictions = modele.predict(X_constante)

    comparaison = pd.DataFrame({
        "Observe": y,
        "Estime": predictions,
        "Residu": y - predictions
    })

    enregistrerTableau(
        comparaison,
        nom + "_predictions.csv"
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(y, predictions, alpha=0.65)

    minimum = min(y.min(), predictions.min())
    maximum = max(y.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        color="red"
    )

    plt.xlabel("Valeurs observées")
    plt.ylabel("Valeurs estimées")
    plt.title("Régression : " + nom)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTATS / (nom + "_graphique.png"),
        dpi=200
    )
    plt.close()

    return modele, donnees


# ============================================================
# QUESTION 1
# TEMPÉRATURE DES VILLES AMÉRICAINES
# ============================================================

print("\nQUESTION 1")

temperature = ouvrirUnFichier("temperature.csv")

variables_temperature = [
    "Latitude",
    "Longitude",
    "Altitude"
]

cible_temperature = "Température_en_janvier"


# ------------------------------------------------------------
# QUESTION 1 - PREMIER TIRET
# Corrélations
# ------------------------------------------------------------

print("\nQuestion 1 - Premier tiret")

correlations = temperature[
    [cible_temperature] + variables_temperature
].corr()

print(correlations)

enregistrerTableau(
    correlations,
    "question_1_correlations.csv"
)


# ------------------------------------------------------------
# QUESTION 1 - DEUXIÈME TIRET
# Statistiques descriptives
# ------------------------------------------------------------

print("\nQuestion 1 - Deuxième tiret")

statistiques = temperature.describe()

print(statistiques)

enregistrerTableau(
    statistiques,
    "question_1_statistiques.csv"
)


# ------------------------------------------------------------
# QUESTION 1 - TROISIÈME TIRET
# Séparation des variables
# ------------------------------------------------------------

print("\nQuestion 1 - Troisième tiret")

villes = temperature["Ville"]

y_temperature = temperature[cible_temperature]
X_temperature = temperature[variables_temperature]

print("Nombre de villes :", len(villes))
print("Variable à expliquer :", cible_temperature)
print("Variables explicatives :", variables_temperature)


# ------------------------------------------------------------
# QUESTION 1 - QUATRIÈME TIRET
# Régression avec Statsmodels
# ------------------------------------------------------------

print("\nQuestion 1 - Quatrième tiret")

modele_temperature, donnees_temperature = calculerRegression(
    temperature,
    cible_temperature,
    variables_temperature,
    "question_1_temperature"
)


# ------------------------------------------------------------
# QUESTION 1 - CINQUIÈME TIRET
# Paramètres de la régression
# ------------------------------------------------------------

print("\nQuestion 1 - Cinquième tiret")

print("Constante :", modele_temperature.params["const"])

for variable in variables_temperature:
    print(
        variable,
        ":",
        modele_temperature.params[variable]
    )

print("R2 :", modele_temperature.rsquared)
print("R2 ajusté :", modele_temperature.rsquared_adj)


# ------------------------------------------------------------
# QUESTION 1 - SIXIÈME TIRET
# Régression avec Scikit-Learn
# ------------------------------------------------------------

print("\nQuestion 1 - Sixième tiret")

regression_sklearn = LinearRegression()

regression_sklearn.fit(
    donnees_temperature[variables_temperature],
    donnees_temperature[cible_temperature]
)

print("Coefficients Scikit-Learn :")

for variable, coefficient in zip(
    variables_temperature,
    regression_sklearn.coef_
):
    print(variable, ":", coefficient)

print("Constante :", regression_sklearn.intercept_)

print(
    "R2 Scikit-Learn :",
    regression_sklearn.score(
        donnees_temperature[variables_temperature],
        donnees_temperature[cible_temperature]
    )
)


# ------------------------------------------------------------
# QUESTION 1 - SEPTIÈME TIRET
# Résultats pour le commentaire
# ------------------------------------------------------------

print("\nQuestion 1 - Septième tiret")

print("R2 :", round(modele_temperature.rsquared, 4))

for variable in variables_temperature:
    p = modele_temperature.pvalues[variable]

    if p < 0.05:
        print(variable, ": significative au seuil de 5 %")
    else:
        print(variable, ": non significative au seuil de 5 %")


# ============================================================
# QUESTION 2
# GÉOMARKETING - CHIFFRE D'AFFAIRES
# ============================================================

print("\nQUESTION 2")

geomarketing = ouvrirUnFichier("geomarketing.csv")

variable_ca = "ca"

variables_selectionnees = [
    "surface_totale",
    "potentiel_Z20",
    "nb_primaire_Z10",
    "nb_primaire_Z20",
    "nb_gsa_Z10",
    "nb_pharmacie_Z5",
    "nb_conc2_Z10",
    "nb_conc2_Z20",
    "P10_POP_Z15",
    "P10_MEN_Z10"
]


# ------------------------------------------------------------
# QUESTION 2 - PREMIER TIRET
# Ouvrir et examiner les données
# ------------------------------------------------------------

print("\nQuestion 2 - Premier tiret")

print("Nombre de magasins :", len(geomarketing))
print("Nombre de colonnes :", len(geomarketing.columns))
print(geomarketing.head())


# ------------------------------------------------------------
# QUESTION 2 - DEUXIÈME TIRET
# Isoler les variables sélectionnées
# ------------------------------------------------------------

print("\nQuestion 2 - Deuxième tiret")

y_ca = geomarketing[variable_ca]
X_ca = geomarketing[variables_selectionnees]

print("Variable à expliquer :", variable_ca)
print("Variables explicatives :")
print(variables_selectionnees)

print("\nStatistiques descriptives :")
print(
    geomarketing[
        [variable_ca] + variables_selectionnees
    ].describe()
)


# ------------------------------------------------------------
# QUESTION 2 - TROISIÈME TIRET
# Régression multiple
# ------------------------------------------------------------

print("\nQuestion 2 - Troisième tiret")

modele_ca, donnees_ca = calculerRegression(
    geomarketing,
    variable_ca,
    variables_selectionnees,
    "question_2_geomarketing"
)


# ------------------------------------------------------------
# QUESTION 2 - QUATRIÈME TIRET
# Analyse des coefficients et p-values
# ------------------------------------------------------------

print("\nQuestion 2 - Quatrième tiret")

print("R2 :", round(modele_ca.rsquared, 4))
print("R2 ajusté :", round(modele_ca.rsquared_adj, 4))

for variable in variables_selectionnees:
    coefficient = modele_ca.params[variable]
    p = modele_ca.pvalues[variable]

    print(
        variable,
        "- coefficient :",
        round(coefficient, 4),
        "- p-value :",
        round(p, 6)
    )


# ============================================================
# BONUS
# COMPARAISON AVEC TOUTES LES VARIABLES
# ============================================================

print("\nBONUS")

# Toutes les variables numériques sauf le chiffre d'affaires
variables_completes = [
    colonne
    for colonne in geomarketing.select_dtypes(
        include="number"
    ).columns
    if colonne != "ca"
]

print(
    "Nombre de variables du modèle complet :",
    len(variables_completes)
)

modele_complet, donnees_completes = calculerRegression(
    geomarketing,
    "ca",
    variables_completes,
    "bonus_geomarketing_complet"
)

# Comparaison des deux modèles
comparaison_modeles = pd.DataFrame({
    "Modele": [
        "Variables sélectionnées",
        "Toutes les variables"
    ],
    "Nombre de variables": [
        len(variables_selectionnees),
        len(variables_completes)
    ],
    "R2": [
        modele_ca.rsquared,
        modele_complet.rsquared
    ],
    "R2 ajuste": [
        modele_ca.rsquared_adj,
        modele_complet.rsquared_adj
    ],
    "AIC": [
        modele_ca.aic,
        modele_complet.aic
    ],
    "BIC": [
        modele_ca.bic,
        modele_complet.bic
    ]
})

print("\nComparaison des modèles :")
print(comparaison_modeles.to_string(index=False))

comparaison_modeles.to_csv(
    RESULTATS / "bonus_comparaison_modeles.csv",
    index=False,
    encoding="utf-8-sig"
)

# Un R2 plus élevé ne signifie pas nécessairement
# que le modèle est meilleur.
# Il faut aussi comparer le R2 ajusté, les p-values,
# et vérifier les problèmes de multicolinéarité.

print("\nSéance 9 terminée !")

