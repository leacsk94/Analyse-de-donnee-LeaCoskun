
# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


# ============================================================
# DOSSIERS ET FONCTIONS
# ============================================================

DOSSIER = Path(__file__).resolve().parent
DATA = DOSSIER / "data"
RESULTATS = DOSSIER / "resultats"

RESULTATS.mkdir(exist_ok=True)


def ouvrirUnFichier(nom):
    return pd.read_csv(DATA / nom)


def tableauDeContingence(nom, donnees):
    return pd.DataFrame(donnees, index=list(nom))


def sommeDesColonnes(tableau):
    return tableau.sum(axis=0).tolist()


def sommeDesLignes(tableau):
    return tableau.sum(axis=1).tolist()


# ============================================================
# OUVERTURE DES DONNEES ELECTORALES
# ============================================================

data = ouvrirUnFichier(
    "resultats-elections-presidentielles-2022-1er-tour.csv"
)

candidats = [
    "Nathalie_ARTHAUD",
    "Fabien_ROUSSEL",
    "Emmanuel_MACRON",
    "Jean_LASSALLE",
    "Marine_LE_PEN",
    "Éric_ZEMMOUR",
    "Jean_Luc_MÉLENCHON",
    "Anne_HIDALGO",
    "Yannick_JADOT",
    "Valérie_PÉCRESSE",
    "Philippe_POUTOU",
    "Nicolas_DUPONT_AIGNAN"
]

tableau = tableauDeContingence(
    data["Libellé_du_département"],
    data[candidats].to_numpy()
)

tableau.columns = candidats

print("Tableau de contingence :")
print(tableau.head())


# ============================================================
# QUESTION 1
# CALCUL DES MARGES
# ============================================================

print("\nQUESTION 1")

marges_colonnes = sommeDesColonnes(tableau)
marges_lignes = sommeDesLignes(tableau)

print("Marges des colonnes :")
for candidat, total in zip(candidats, marges_colonnes):
    print(candidat, ":", total)

print("\nPremières marges des lignes :")
for departement, total in zip(
    tableau.index[:10], marges_lignes[:10]
):
    print(departement, ":", total)

# Tableau avec les totaux des lignes et des colonnes
tableau_marges = tableau.copy()

tableau_marges["Total"] = marges_lignes

ligne_totaux = pd.DataFrame(
    [tableau_marges.sum(axis=0)],
    index=["TOTAL"]
)

tableau_marges = pd.concat(
    [tableau_marges, ligne_totaux]
)

tableau_marges.to_csv(
    RESULTATS / "question_1_tableau_marges.csv",
    encoding="utf-8-sig"
)

print("Tableau des marges enregistré")


# ============================================================
# QUESTION 2
# VERIFICATION DES TOTAUX
# ============================================================

print("\nQUESTION 2")

total_colonnes = sum(marges_colonnes)
total_lignes = sum(marges_lignes)

print("Total des colonnes :", total_colonnes)
print("Total des lignes :", total_lignes)

if total_colonnes == total_lignes:
    print("Les deux totaux sont identiques.")
else:
    print("Attention : les deux totaux sont différents.")


# ============================================================
# QUESTION 3
# TEST DU CHI2 D'INDEPENDANCE
# ============================================================

print("\nQUESTION 3")

chi2, p_value, ddl, effectifs_attendus = (
    scipy.stats.chi2_contingency(tableau)
)

print("Chi2 :", chi2)
print("p-value :", p_value)
print("Degrés de liberté :", ddl)

if p_value < 0.05:
    print("On rejette l'hypothèse d'indépendance.")
    print("Il existe une liaison statistique entre")
    print("le département et le choix du candidat.")
else:
    print("On ne rejette pas l'hypothèse d'indépendance.")


# ============================================================
# QUESTION 4
# INTENSITE DE LIAISON : PHI2 DE PEARSON
# ============================================================

print("\nQUESTION 4")

effectif_total = tableau.to_numpy().sum()

phi2 = chi2 / effectif_total

print("Effectif total :", effectif_total)
print("Phi2 de Pearson :", phi2)

# Le Phi2 mesure l'écart à l'indépendance.
# Plus sa valeur est élevée, plus l'écart est important.
# Il ne doit pas être interprété comme un coefficient
# de corrélation compris entre -1 et 1.

# Calcul complémentaire : V de Cramer
nombre_lignes, nombre_colonnes = tableau.shape

v_cramer = np.sqrt(
    phi2 / min(nombre_lignes - 1, nombre_colonnes - 1)
)

print("V de Cramer :", v_cramer)


# ============================================================
# QUESTION 5
# SYNTHESE DES RESULTATS
# ============================================================

print("\nQUESTION 5")

print("Chi2 :", round(chi2, 2))
print("Phi2 :", round(phi2, 6))
print("V de Cramer :", round(v_cramer, 6))

# Le commentaire détaillé sera rédigé dans le rapport.


# ============================================================
# BONUS 1
# ANALYSE ANOVA
# ============================================================

print("\nBONUS 1 - ANOVA")

echantillons = ouvrirUnFichier(
    "Echantillonnage-100-Echantillons.csv"
)

colonnes = ["Pour", "Contre", "Sans opinion"]

echantillons = echantillons[colonnes].apply(
    pd.to_numeric, errors="coerce"
)

echantillons = echantillons.dropna()

print("Nombre d'échantillons :", len(echantillons))

print("\nMoyennes :")
print(echantillons.mean())

# ANOVA à un facteur
anova = scipy.stats.f_oneway(
    echantillons["Pour"],
    echantillons["Contre"],
    echantillons["Sans opinion"]
)

print("\nStatistique F :", anova.statistic)
print("p-value :", anova.pvalue)

if anova.pvalue < 0.05:
    print("Les moyennes ne sont pas toutes égales.")
else:
    print("Aucune différence significative détectée.")

# Enregistrement du tableau descriptif
echantillons.describe().to_csv(
    RESULTATS / "bonus_1_statistiques_anova.csv",
    encoding="utf-8-sig"
)

# Graphique des distributions
plt.figure(figsize=(9, 5))

plt.boxplot(
    [echantillons[c].to_numpy() for c in colonnes],
    tick_labels=colonnes
)

plt.ylabel("Effectifs")
plt.title("ANOVA - Comparaison des trois catégories")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTATS / "bonus_1_anova.png",
    dpi=200
)

plt.close()

print("Graphique ANOVA enregistré")


# ============================================================
# BONUS 2
# ANALYSE FACTORIELLE DES CORRESPONDANCES (AFC)
# ============================================================

print("\nBONUS 2 - AFC")

# Matrice des effectifs
N = tableau.to_numpy(dtype=float)

# Tableau des fréquences
P = N / N.sum()

# Masses des lignes et des colonnes
r = P.sum(axis=1)
c = P.sum(axis=0)

# Matrice des écarts à l'indépendance standardisés
S = (
    (P - np.outer(r, c))
    / np.sqrt(np.outer(r, c))
)

# Décomposition en valeurs singulières
U, valeurs_singulieres, Vt = np.linalg.svd(
    S, full_matrices=False
)

# Inerties des axes
inerties = valeurs_singulieres ** 2

inertie_totale = inerties.sum()

pourcentages = 100 * inerties / inertie_totale

print("Inertie totale :", inertie_totale)
print("Inertie de l'axe 1 :", pourcentages[0], "%")
print("Inertie de l'axe 2 :", pourcentages[1], "%")

# Coordonnées principales des lignes
coord_lignes = (
    U[:, :2] * valeurs_singulieres[:2]
) / np.sqrt(r[:, None])

# Coordonnées principales des colonnes
coord_colonnes = (
    Vt.T[:, :2] * valeurs_singulieres[:2]
) / np.sqrt(c[:, None])

# Enregistrement des coordonnées
resultats_lignes = pd.DataFrame(
    coord_lignes,
    columns=["Axe 1", "Axe 2"],
    index=tableau.index
)

resultats_colonnes = pd.DataFrame(
    coord_colonnes,
    columns=["Axe 1", "Axe 2"],
    index=tableau.columns
)

resultats_lignes.to_csv(
    RESULTATS / "bonus_2_afc_departements.csv",
    encoding="utf-8-sig"
)

resultats_colonnes.to_csv(
    RESULTATS / "bonus_2_afc_candidats.csv",
    encoding="utf-8-sig"
)

# Graphique AFC
plt.figure(figsize=(12, 8))

plt.scatter(
    coord_lignes[:, 0],
    coord_lignes[:, 1],
    alpha=0.45,
    label="Départements"
)

plt.scatter(
    coord_colonnes[:, 0],
    coord_colonnes[:, 1],
    marker="^",
    s=75,
    label="Candidats"
)

# Étiquettes des candidats
for i, candidat in enumerate(candidats):
    plt.annotate(
        candidat.replace("_", " "),
        (coord_colonnes[i, 0], coord_colonnes[i, 1]),
        fontsize=8
    )

plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(0, color="gray", linewidth=0.8)

plt.xlabel(f"Axe 1 ({pourcentages[0]:.1f} %)")
plt.ylabel(f"Axe 2 ({pourcentages[1]:.1f} %)")

plt.title(
    "AFC - Résultats présidentiels 2022 par département"
)

plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    RESULTATS / "bonus_2_afc.png",
    dpi=200
)

plt.close()

# Enregistrement des inerties
pd.DataFrame({
    "Axe": np.arange(1, len(inerties) + 1),
    "Inertie": inerties,
    "Pourcentage": pourcentages
}).to_csv(
    RESULTATS / "bonus_2_inerties.csv",
    index=False
)

print("Graphique AFC enregistré")
print("Coordonnées et inerties enregistrées")

print("\nSéance 7 terminée !")

