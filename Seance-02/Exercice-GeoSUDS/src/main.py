#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def listedesterritoires(territoiretest, exportationUSD, importationUSD):
    listedesterritoires = [[territoiretest[0], 0]]
    pos = 0
    for element in range(0,len(territoiretest)):
        if territoiretest[element] != listedesterritoires[pos][0]:
            listedesterritoires.append([territoiretest[element], element])
            pos += 1
    lignes = []
    for element in range(1,len(listedesterritoires)):
        lignes.append(listedesterritoires[element][1] - 1)
    lignes.append(len(territoiretest))
    liste = []
    for element in range(0,len(listedesterritoires)):
        liste.append([listedesterritoires[element][0], listedesterritoires[element][1], lignes[element]])
    totalparterritoire = []
    for element in range(0,len(liste)):
        totalparterritoire.append([liste[element][0], exportationUSD[liste[element][1]:liste[element][2]].sum(), importationUSD[liste[element][1]:liste[element][2]].sum()])
    return totalparterritoire

def nettoyage(colonne):
    colonne2 = []
    for element in colonne:
        if element == 'Sans objet' or element =='-':
            colonne2.append(float(0))
        else:
            colonne2.append(float(element))
    return colonne2

def getliste(territoiretest):
    listedesterritoires = [[territoiretest[0], 0]]
    pos = 0
    for element in range(0,len(territoiretest)):
        if territoiretest[element] != listedesterritoires[pos][0]:
            listedesterritoires.append([territoiretest[element], element])
            pos += 1
    lignes = []
    for element in range(1,len(listedesterritoires)):
        lignes.append(listedesterritoires[element][1] - 1)
    lignes.append(len(territoiretest))
    liste = []
    for element in range(0,len(listedesterritoires)):
        liste.append([listedesterritoires[element][0], listedesterritoires[element][1], lignes[element]])
    return liste

# Question 4
print("Question 4")
# Source des données : https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
with open("./data/Afrique-2024.csv", "r", encoding="utf-8") as fichier:
    contenu = pd.read_csv(fichier)
# Question 5
print("Question 5")
print(len(contenu))

# Question 6
print("Question 6")
print(len(contenu))
print(len(contenu.columns))

# Question 7
print("Question 7")
print(contenu.head())

# Question 8
print("Question 8")
print(contenu.dtypes)

# Contenu des colonnes
# "year" : année
# "origin_id" : territoire de référence
# "dest_id" : territoire partenaire
# "Dist_VO (km)" : distance intercentroïde entre les deux territoires
# "hs92_Section_id" : code Section du produit
# "hs92_HS02_id" : code 2-digit du produit
# "hs92_HS04_id" : code 4-digit du produit
# "hs92_HS06_id" : code 6-digit du produit
# "export_val" : valeur des exportations du territoire de référence vers le territoire partenaire en dollars courants
# "import_val" : valeur des importations du territoire partenaire vers le territoire de référence en dollars courants
# "export_ton" : valeur des exportations du territoire de référence vers le territoire partenaire en tonnes
# "import_ton" : valeur des importations du territoire partenaire vers le territoire de référence en tonnes

# Question 9
print("Question 9")
print(contenu.isna().sum())

# Question 10
print("Question 10")
print(contenu.describe())

# Question 11
print("Question 11")
# Appliquer la fonction listedesterritoires(...) sur vos colonnes ici
print(listedesterritoires(contenu["dest_id"], contenu["export_val"], contenu["import_val"]))

# Questions 12 et 14
print("Questions 12 et 14")
# Application la fonction nettoyage(...) sur vos colonnes ici
distance = nettoyage(contenu["Dist_VO (km)"])
export_val_2 = nettoyage(contenu["export_val"])
import_val_2 = nettoyage(contenu["import_val"])
export_ton_2 = nettoyage(contenu["export_ton"])
import_ton_2 = nettoyage(contenu["import_ton"])
liste = getliste(contenu["origin_id"])

# Appliquer la fonction :
# liste = getliste(...)

# Mettre le code nouveau DataFrame ici et appeler la liste donnees2
donnees2titre = ["distance", "export_val_2", "import_val_2", "export_ton_2", "import_ton_2"]
donnees2 = pd.DataFrame({
    donnees2titre[0]: distance,
    donnees2titre[1]: export_val_2,
    donnees2titre[2]: import_val_2,
    donnees2titre[3]: export_ton_2,
    donnees2titre[4]: import_ton_2
})

# Compléter la boucle à partir des différents éléments précédents. Le nom des listes est celle proposée dans les commentaires.
parametres = []

quartiles = []
deciles = []
for element in range(0,len(donnees2.columns)):
     parametres2 = []
     distanceinterquartile = []
     distanceinterdecile = []
     for element2 in range(0,len(liste)):
         codeiso = liste[element2][0]
         data2 = donnees2.iloc[liste[element2][1]:liste[element2][2],element]

         moyenne = data2.mean()
         mediane = data2.median()
         mode = data2.mode().iloc[0]
         ecarttype = data2.std()
         ecartabsolumoyen = stats.median_abs_deviation(data2)
         etendue =  data2.max() - data2.min()
         parametres2.append([codeiso, moyenne, mediane, mode, ecarttype, ecartabsolumoyen, etendue])

         quartile = data2.quantile([0.25, 0.75])
         distanceinterquartile.append([codeiso, (quartile[0.75] - quartile[0.25]).round(decimals=2)])
         decile = data2.quantile([0.1, 0.9])
         distanceinterdecile.append([codeiso, (decile[0.9] - decile[0.1]).round(decimals=2)])
     parametres.append(parametres2)
     quartiles.append(distanceinterquartile)
     deciles.append(distanceinterdecile)

# Question 13
print("Question 13")
print(parametres)

# Question 14
# print("Question 14")

# Question 15
print("Question 15")
for element in donnees2.columns:
    plt.boxplot(donnees2[element])
    plt.title(element)
    plt.savefig("img/boxplot_" + element + ".png")
    plt.close()

# Question 16
print("Question 16")
distance = nettoyage(contenu["Dist_VO (km)"])

classes = [0, 0, 0, 0, 0, 0, 0, 0]

for element in distance:
    if 0 < element <= 2500:
        classes[0] += 1
    elif 2500 < element <= 5000:
        classes[1] += 1
    elif 5000 < element <= 7500:
        classes[2] += 1
    elif 7500 < element <= 10000:
        classes[3] += 1
    elif 10000 < element <= 12500:
        classes[4] += 1
    elif 12500 < element <= 15000:
        classes[5] += 1
    elif 15000 < element <= 17500:
        classes[6] += 1
    elif 17500 < element <= 20000:
        classes[7] += 1

print(classes)
# Question bonus
print("Question bonus")
country = pd.read_excel("data/country_names.xls")
print(country.head())
print(country.columns)
print(country[["id_3char", "name"]].head())

for element in range(0, len(liste)):
    codeiso = liste[element][0]

    nom = country.loc[country["id_3char"] == codeiso, "name"]

    if len(nom) > 0:
        nom = nom.iloc[0]
    else:
        nom = codeiso

    valeurs = [
        contenu["export_val"][liste[element][1]:liste[element][2]].sum(),
        contenu["import_val"][liste[element][1]:liste[element][2]].sum()
    ]

    plt.pie(valeurs, labels=["Export", "Import"])
    plt.title(nom)
    plt.savefig("img2/" + codeiso + ".png")
    plt.close()
# Export des listes en CSV et Excel
pd.DataFrame(parametres).to_csv("csv/parametres.csv", index=False)
pd.DataFrame(parametres).to_excel("xlsx/parametres.xlsx", index=False)

pd.DataFrame(quartiles).to_csv("csv/quartiles.csv", index=False)
pd.DataFrame(quartiles).to_excel("xlsx/quartiles.xlsx", index=False)

pd.DataFrame(deciles).to_csv("csv/deciles.csv", index=False)
pd.DataFrame(deciles).to_excel("xlsx/deciles.xlsx", index=False)