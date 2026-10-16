
# coding: utf-8
"""Séance 8 GEOINT : ACP, ACM et bonus CAH."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import prince


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTATS = BASE / "resultats"
IMG = BASE / "img"

RESULTATS.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)


def exporter(tableau, nom):
    tableau.to_csv(RESULTATS / nom, encoding="utf-8-sig")
    print("Enregistré :", nom)


def graphique_individus(coord, labels, titre, fichier, max_labels=35):
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(
        coord.iloc[:, 0],
        coord.iloc[:, 1],
        s=24,
        alpha=0.7
    )

    for i in range(min(len(coord), max_labels)):
        ax.annotate(
            str(labels[i]),
            (coord.iloc[i, 0], coord.iloc[i, 1]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points"
        )

    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)

    ax.set(
        xlabel="Facteur 1",
        ylabel="Facteur 2",
        title=titre
    )

    fig.tight_layout()
    fig.savefig(IMG / fichier, dpi=160)
    plt.close(fig)


# ==================================================
# QUESTION 1 : ANALYSE EN COMPOSANTES PRINCIPALES
# ==================================================

print("\nQUESTION 1 : Analyse en composantes principales")


# Question 1a et 1b
# Récupérer les données SIPRI et isoler les variables

sipri = pd.read_csv(DATA / "SIPRI-1949-2025.csv")

codes = sipri["Code ISO"].fillna("Inconnu").astype(str)

etats = (
    sipri["État"]
    .fillna(sipri["Country"])
    .fillna(codes)
    .astype(str)
)

numeriques = sipri.drop(
    columns=["Code ISO", "Country", "État", "Continent"]
)

numeriques = numeriques.apply(
    pd.to_numeric,
    errors="coerce"
)

print(
    "Question 1a-b :",
    len(sipri),
    "États et",
    numeriques.shape[1],
    "années"
)


# Question 1c
# Remplacer les valeurs manquantes et centrer-réduire

numeriques = numeriques.fillna(0)

X = StandardScaler().fit_transform(numeriques)

print("Question 1c : données centrées-réduites")


# Question 1d
# Réaliser une ACP avec 77 facteurs

n_facteurs = numeriques.shape[1]

pca = PCA(n_components=n_facteurs)

coordonnees = pca.fit_transform(X)

facteurs = [
    f"F{i}"
    for i in range(1, n_facteurs + 1)
]

print("Question 1d : ACP calculée sur", n_facteurs, "facteurs")

print(
    "Variance expliquée par les deux premiers facteurs :",
    np.round(pca.explained_variance_ratio_[:2] * 100, 2),
    "%"
)


# Question 1e
# Variance expliquée, pourcentage et valeurs propres

variance = pd.DataFrame(
    {
        "variance_expliquee": pca.explained_variance_,
        "pourcentage": pca.explained_variance_ratio_ * 100,
        "pourcentage_cumule": (
            np.cumsum(pca.explained_variance_ratio_) * 100
        ),
        "valeur_propre": pca.explained_variance_
    },
    index=facteurs
)

exporter(
    variance,
    "question_1e_variances_valeurs_propres.csv"
)


# Question 1f et 1g
# Coordonnées des individus et graphique

coord = pd.DataFrame(
    coordonnees,
    columns=facteurs,
    index=codes
)

coord.insert(0, "État", etats.to_numpy())

exporter(
    coord,
    "question_1f_coordonnees_individus.csv"
)

graphique_individus(
    coord[facteurs],
    etats.tolist(),
    "ACP SIPRI : dépenses militaires (1949-2025)",
    "question_1g_mapping_individus.png"
)

print("Question 1g : graphique des individus créé")


# Question 1h
# Contribution des individus aux facteurs

valeurs_propres = pca.explained_variance_

denom = (len(sipri) - 1) * valeurs_propres

contributions = np.divide(
    coordonnees ** 2,
    denom[np.newaxis, :],
    out=np.zeros_like(coordonnees),
    where=denom[np.newaxis, :] > 1e-12
) * 100

contrib = pd.DataFrame(
    contributions,
    index=codes,
    columns=[
        f"Contribution_{f}"
        for f in facteurs
    ]
)

exporter(
    contrib,
    "question_1h_contributions_individus.csv"
)


# Qualité de représentation : cosinus carrés

distance2 = np.sum(
    coordonnees ** 2,
    axis=1,
    keepdims=True
)

cos2 = np.divide(
    coordonnees ** 2,
    distance2,
    out=np.zeros_like(coordonnees),
    where=distance2 > 1e-12
)

qualite = pd.DataFrame(
    cos2,
    index=codes,
    columns=[
        f"Cos2_{f}"
        for f in facteurs
    ]
)

exporter(
    qualite,
    "question_1h_cosinus_carres_individus.csv"
)

print(
    "Question 1h : un cos² proche de 1 indique "
    "une bonne représentation sur le facteur."
)


# Question 1i
# Coordonnées des variables et cercle des corrélations

coords_var = (
    pca.components_.T * np.sqrt(valeurs_propres)
)

var_df = pd.DataFrame(
    coords_var,
    index=numeriques.columns,
    columns=facteurs
)

exporter(
    var_df,
    "question_1i_coordonnees_variables.csv"
)

fig, ax = plt.subplots(figsize=(10, 10))

ax.add_patch(
    Circle(
        (0, 0),
        1,
        fill=False,
        color="gray"
    )
)

for annee, x, y in zip(
    numeriques.columns,
    coords_var[:, 0],
    coords_var[:, 1]
):
    ax.arrow(
        0,
        0,
        x,
        y,
        head_width=0.015,
        alpha=0.35,
        length_includes_head=True
    )

    ax.text(x, y, str(annee), fontsize=6)

ax.set(
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
    aspect="equal",
    xlabel="Facteur 1",
    ylabel="Facteur 2",
    title="ACP SIPRI : cercle des corrélations"
)

ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)

fig.tight_layout()

fig.savefig(
    IMG / "question_1i_cercle_correlations.png",
    dpi=160
)

plt.close(fig)

print("Question 1i : cercle des corrélations créé")


# ==================================================
# QUESTION 2 : ANALYSE DES CORRESPONDANCES MULTIPLES
# ==================================================

print("\nQUESTION 2 : Analyse des correspondances multiples")


# Question 2a et 2b
# Charger les données sur les races de chiens

chiens = pd.read_csv(DATA / "chiens.csv")

races = (
    chiens["Race"]
    .fillna("Race inconnue")
    .astype(str)
)

colonnes = [
    "Taille",
    "Poids",
    "Vitesse",
    "Intelligence",
    "Affection",
    "Agressivité",
    "Fonction",
    "Origine"
]

variables = (
    chiens[colonnes]
    .fillna("Non renseigné")
    .astype(str)
)

print(
    "Question 2a-b :",
    len(chiens),
    "races et",
    len(colonnes),
    "variables"
)


# Question 2c
# Construire le tableau disjonctif complet

tdc = pd.get_dummies(
    variables,
    dtype=int
)

tdc.index = races

exporter(
    tdc,
    "question_2c_tableau_disjonctif_complet.csv"
)


# Question 2d
# Réaliser une ACM avec huit facteurs

acm = prince.MCA(
    n_components=8,
    one_hot=False,
    engine="scipy",
    random_state=42
)
acm = acm.fit(tdc)

print("Question 2d : ACM calculée sur huit facteurs")


# Question 2e
# Calculer les valeurs propres

vp_acm = np.asarray(acm.eigenvalues_)

valeurs_acm = pd.DataFrame(
    {"valeur_propre": vp_acm},
    index=[
        f"F{i}"
        for i in range(1, len(vp_acm) + 1)
    ]
)

exporter(
    valeurs_acm,
    "question_2e_valeurs_propres.csv"
)


# Question 2f
# Coordonnées des lignes et des colonnes

lignes = acm.row_coordinates(tdc)

colonnes_acm = acm.column_coordinates(tdc)

lignes.columns = [
    f"F{i}"
    for i in range(1, lignes.shape[1] + 1)
]

colonnes_acm.columns = [
    f"F{i}"
    for i in range(1, colonnes_acm.shape[1] + 1)
]

exporter(
    lignes,
    "question_2f_coordonnees_lignes.csv"
)

exporter(
    colonnes_acm,
    "question_2f_coordonnees_colonnes.csv"
)


# Question 2g
# Mapping des deux premiers facteurs

graphique_individus(
    lignes,
    races.tolist(),
    "ACM : races de chiens, facteurs 1 et 2",
    "question_2g_mapping_races.png",
    max_labels=len(races)
)

fig, ax = plt.subplots(figsize=(12, 9))

ax.scatter(
    colonnes_acm.iloc[:, 0],
    colonnes_acm.iloc[:, 1],
    s=20
)

for nom, x, y in zip(
    colonnes_acm.index,
    colonnes_acm.iloc[:, 0],
    colonnes_acm.iloc[:, 1]
):
    ax.annotate(
        str(nom),
        (x, y),
        fontsize=7
    )

ax.axhline(0, color="gray", lw=0.7)
ax.axvline(0, color="gray", lw=0.7)

ax.set(
    xlabel="Facteur 1",
    ylabel="Facteur 2",
    title="ACM : modalités des variables"
)

fig.tight_layout()

fig.savefig(
    IMG / "question_2g_mapping_modalites.png",
    dpi=160
)

plt.close(fig)

print("Question 2g : graphiques créés")


# Question 2h
# Qualité de représentation des lignes et colonnes

cos_lignes = acm.row_cosine_similarities(tdc)

cos_colonnes = acm.column_cosine_similarities(tdc)

exporter(
    cos_lignes,
    "question_2h_cos2_lignes.csv"
)

exporter(
    cos_colonnes,
    "question_2h_cos2_colonnes.csv"
)

print(
    "Question 2h : un cos² élevé indique "
    "une meilleure représentation."
)


# ==================================================
# BONUS : CLASSIFICATION ASCENDANTE HIÉRARCHIQUE
# ==================================================

print("\nBONUS : Classification ascendante hiérarchique")

# CAH de Ward sur les premiers axes de l'ACP

axes_cah = min(10, n_facteurs)

Z = linkage(
    coordonnees[:, :axes_cah],
    method="ward"
)

fig, ax = plt.subplots(figsize=(17, 9))

dendrogram(
    Z,
    labels=etats.tolist(),
    leaf_rotation=90,
    leaf_font_size=5,
    ax=ax
)

ax.set(
    title=(
        f"CAH de Ward sur les "
        f"{axes_cah} premiers axes de l'ACP"
    ),
    ylabel="Distance de fusion"
)

fig.tight_layout()

fig.savefig(
    IMG / "bonus_cah_sipri.png",
    dpi=160
)

plt.close(fig)

print("\nSéance 8 terminée !")
print("Tableaux :", RESULTATS)
print("Graphiques :", IMG)
