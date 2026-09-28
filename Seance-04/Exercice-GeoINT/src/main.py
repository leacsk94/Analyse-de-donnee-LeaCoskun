#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

# C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
# Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
# Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

# Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r", encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

# Question 1 : Théorie de l'échantillonnage (intervalles de fluctuation)
# L'échantillonnage se base sur la répétitivité.
print("Question 1")
print("Résultat sur le calcul d'un intervalle de fluctuation")
echantillons = ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv")
print(echantillons.head())
moyennes = []

for colonne in echantillons.columns:
    moyenne = round(echantillons[colonne].mean())
    moyennes.append(moyenne)

print(moyennes)
total_moyennes = sum(moyennes)

frequences = []
for moyenne in moyennes:
    frequences.append(round(moyenne / total_moyennes, 2))

print("Fréquences des échantillons :", frequences)
population = [852, 911, 422]
total_population = sum(population)

frequences_population = []
for valeur in population:
    frequences_population.append(round(valeur / total_population, 2))

print("Fréquences de la population mère :", frequences_population)
zc = 1.96

for frequence in frequences:
    borne_inf = frequence - zc * math.sqrt((frequence * (1 - frequence)) / total_moyennes)
    borne_sup = frequence + zc * math.sqrt((frequence * (1 - frequence)) / total_moyennes)

    print("Intervalle de fluctuation :", round(borne_inf, 2), round(borne_sup, 2))

# Question 2 : Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Question 2")
print("Résultat sur le calcul d'un intervalle de confiance")
premier_echantillon = list(echantillons.iloc[0])

total_echantillon = sum(premier_echantillon)

frequences_echantillon = []
for valeur in premier_echantillon:
    frequences_echantillon.append(round(valeur / total_echantillon, 2))

print("Premier échantillon :", premier_echantillon)
print("Effectif total :", total_echantillon)
print("Fréquences :", frequences_echantillon)
for frequence in frequences_echantillon:
    borne_inf = frequence - zc * math.sqrt((frequence * (1 - frequence)) / total_echantillon)
    borne_sup = frequence + zc * math.sqrt((frequence * (1 - frequence)) / total_echantillon)

    print("Intervalle de confiance :", round(borne_inf, 2), round(borne_sup, 2))


# Question 3 : Théorie de la décision (tests d'hypothèse)
# La décision se base sur la notion de risques alpha et bêta.
# Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Question 3")
print("Théorie de la décision")
test1 = ouvrirUnFichier("data/Loi-normale-Test-1.csv")
test2 = ouvrirUnFichier("data/Loi-normale-Test-2.csv")

resultat_test1 = scipy.stats.shapiro(test1.iloc[:, 0])
resultat_test2 = scipy.stats.shapiro(test2.iloc[:, 0])

print("Test 1 :", resultat_test1)
print("Test 2 :", resultat_test2)


# Question bonus
print("Question bonus")
print("La distribution non normale est le Test 2.")
print("Elle correspond à une loi de Zipf.")
