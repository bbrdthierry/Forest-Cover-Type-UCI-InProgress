#!/usr/bin/env python
# coding: utf-8

# # <center> Forest Cover Type </center>

# Groupe : BOMBARD Thierry, DIALLO Ibrahima, MARC Jordan

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Présentation des données

# Le jeu de données Forest Cover Type provient de plusieurs sources :
# 
# - Remote Sensing and GIS Program
# - Department of Forest Sciences
# - College of Natural Resources
# - Colorado State University
# - Fort Collins, CO 80523
# 
# La récolte de ces données a été effectuée dans 4 différentes zones sauvages situées dans la forêt nationale Roosevelt (dans une parcelle de 30x30 mètres chacunes), au nord du Colorado.
# Ces zones représentent des forêts avec des perturbations anthropiques minimales, de telles sorte que les types de couvert forestier soient à priori le résultat d'un processus écologique plutôt que de pratiques de gestions forestières.
# Ces données traitent donc la caractérisation de plusieurs types de couvert forestier, selon une panoplie d'éléments cartographiques.
# 
# Définitions : 
# - Un couvert forestier désigne l'ensemble formé par les cimes des arbres d'une forêt.
# - Une perturbation anthropique est une perturbation écologique causée par l'Homme.
# 
# Nous avons listé les 7 différents types de couverts forestiers avec pour chaucun de ces types leurs correspondances numériques renseignés dans notre jeu de données :
# 
#         Spruce/Fir (1)
#         Lodgepole Pine (2)
#         Ponderosa Pine (3)
#         Cottonwood/Willow (4)
#         Aspen (5)
#         Douglas-fir (6)
#         Krummholz (7)
# 
# Notre dataset initial contient 15120 observations pour un total de 56 attributs.
# Toutes les données sont numériques et il n'existe aucune valeur manquante.
# 
# Significations des attributs :
# 
#         'Elevation' : élévation en mètres.
#         'Aspect' : l'aspect en degrés azimuts.
#         'Slope' : pente en degrés.
#         'Horizontal_Distance_To_Hydrology' : distance horizontale à la surface d'eau la plus proche.
#         'Vertical_Distance_To_Hydrology' : distance verticale à la surface d'eau la plus proche.
#         'Horizontal_Distance_To_Roadways' : distance horizontale à la chaussée la plus proche.
#         'Hillshade_9am' (indice allant de 0 à 255) : indice d'ombre portée à 9h au solstice d'été.
#         'Hillshade_Noon' (indice allant de 0 à 255) : indice d'ombre portée à midi, au solstice d'été.
#         'Hillshade_3pm' (indice allant de 0 à 255) : indice d'ombre portée à 15h, au solstice d'été.
#         'Horizontal_Distance_To_Fire_Points' : distance horizontale au point de départ du feu de forêt le plus proche.
#         'Wilderness_Area1' (variable binaire, 0 = absence, 1 = présence) : zone sauvage Rawah.
#         'Wilderness_Area2' (variable binaire, 0 = absence, 1 = présence) : zone sauvage Neota.
#         'Wilderness_Area3' (variable binaire, 0 = absence, 1 = présence) : zone sauvage Comanche Peak.
#         'Wilderness_Area4' (variable binaire, 0 = absence, 1 = présence) : zone sauvage Cache la Poudre.
#         'Soil_Type1 (jusqu'à Soil_Type40, variable binaire, 0 = absence, 1 = présence) : types de sols.
#         'Cover_Type' (valeurs numériques allant de 1 à 7) : couverts forestiers.
# 
# Liste des différents types de sols :
# 
#         1 Famille de la cathédrale - Complexe d’affleurements rocheux, extrêmement pierreux.
#         2 Vanet - Complexe familial Ratake, très pierreux.
#         3 Haploborolis - Complexe d’affleurements rocheux.
#         4 Famille Ratake - Complexe d’affleurements rocheux.
#         5 Famille Vanet - Complexe complexe d’affleurements rocheux.
#         6 Vanet - Familles Wetmore - Complexe d’affleurements rocheux, pierreux.
#         7 Famille gothique.
#         8 Superviseur - Complexe de familles souples.
#         9 Famille Troutville, très pierreuse.
#         10 Bullwark - Familles Catamount - Complexe d’affleurements rocheux.
#         11 Bullwark - Familles Catamount - Complexe rocheux, rocailleux.
#         12 Famille Legault - Complexe rocheux, pierreux.
#         13 Famille Catamount - Rock land - Complexe familial Bullwark, en ruines.
#         14 Argiborolis pachic - Complexe Aquolis.
#         15 non spécifié dans l’enquête USFS Soil and ELU.
#         16 Cryaquolis - Complexe cryoborolique.
#         17 Famille Gateview - Cryaquolis
#         18 Famille Rogert, très pierreuse.
#         19 Cryaquolis typique - Complexe borohémiste.
#         20 Typic Cryaquepts - Typic Cryaquolls complex.
#         21 Cryaquolls typiques - Famille Leighcan, complexe de substrat de till.
#         22 Famille Leighcan, substrat de till, fort semblable à un bloc.
#         23 Famille Leighcan, substrat de till - Complexe typique de Cryaquolls.
#         24 Famille Leighcan, extrêmement pierreuse.
#         25 Famille Leighcan, chaude, extrêmement pierreuse.
#         26 Granile - Complexe familial Catamount, très pierreux.
#         27 Famille Leighcan, chaud - Complexe d’affleurements rocheux, extrêmement pierreux.
#         28 Famille Leighcan - Complexe d’affleurements rocheux, extrêmement pierreux.
#         29 Como - Complexe familial Legault, extrêmement pierreux.
#         30 Famille Como - Rocher - Complexe familial Legault, extrêmement pierreux.
#         31 Leighcan - Catamount familles complexe, extrêmement pierreux.
#         32 Famille Catamount - Affleurement rocheux - Complexe familial Leighcan
#         33 Leighcan - Familles Catamount - Complexe d’affleurements rocheux, extrêmement pierreux.
#         34 Cryorthents - Complexe rocheux, extrêmement pierreux.
#         35 Cryumbrepts - Affleurement rocheux - Complexe Cryaquepts.
#         36 Famille Bross - Terrain rocheux - Complexe Cryumbrepts, extrêmement pierreux.
#         37 Affleurement rocheux - Cryumbrepts - Complexe de cryorthents, extrêmement pierreux.
#         38 Leighcan - Familles Moran - Complexe de Cryaquolls, extrêmement pierreux.
#         39 Famille Moran - Cryorthents - Complexe familial Leighcan, extrêmement pierreux.
#         40 Famille Moran - Cryorthents - Complexe rocheux, extrêmement pierreux.
# 
# 

# In[2]:


df = pd.read_csv('train.csv')
df


# ## Quelques manipulations avec pandas et numpy

# In[3]:


# Extraction du jeu de donnés en csv
df.to_csv('df_csv.csv', encoding='utf-8')


# In[4]:


# Trier par ordre décroissant les valeurs d'Elevation
df.sort_values(by='Elevation', ascending=False)


# In[5]:


# La somme des pentes par zone sauvage 1, indexé par les types de couverts forestiers.
table1 = pd.pivot_table(df, values='Slope', index=['Cover_Type'],
                    columns=['Wilderness_Area1'], aggfunc=np.sum)
table1


# In[6]:


table1.isna() # Pour détecter les valeurs manquantes pour chaque valeur de la zone sauvage 1


# In[7]:


# Génération de nombre allant de 3 à 33 avec un pas de 2
np.arange(3, 33, 2)


# In[8]:


# Utilisation de pd.where
arange1 = pd.Series(np.arange(3, 33, 2))
arange1.where(arange1 > 10)


# In[9]:


# Utilisation de sample pour extraire certains éléments du dataframe, avec un seed de 1
sample1 = df['Slope'].sample(n=3, random_state=1)
sample1


# In[10]:


# Sélection des lignes qui contiennent '77' avec filter
df.filter(like='77', axis=0)


# In[11]:


# Nous pouvons utiliser aussi filter pour sélectionner que certaines colonnes de notre dataframe
df.filter(items=['Elevation', 'Aspect'])


# ## Exploration des données

# In[12]:


df.dtypes # toutes les valeurs sont numériques


# In[13]:


pd.set_option('display.max_columns', None) # pour voir l'ensemble des colonnes
df.describe()


# Soil_Type7 et Soil_Type15 sont des colonnes remplies de 0. 
# La colonne Id ne nous est pas utile étant donné que nous utilisons l'indexation de pandas.

# In[14]:


df = df.drop(['Id', 'Soil_Type7', 'Soil_Type15'], axis = 1) # suppression des colonnes Id, Soil_Type7 et Soil_Type15


# In[15]:


df["Cover_Type"].unique()


# 15120 observations et 53 variables explicatives, après une première suppression des colonnes inutiles.

# ## Valeurs nulles

# In[16]:


# Comptage des valeurs nulles/manquantes par colonnes
df.isnull().sum() # df.isna().sum() fonctionne aussi


# ## Equilibre de la variable à prédire

# In[17]:


df.loc[:,['Cover_Type', 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']].groupby(by='Cover_Type').count()


# Le nombre de couverts forestiers (variable à prédire) est équilibré. Pour chaque type nous avons 2160 lignes dans notre dataset.

# ## Coefficients d'asymétrie

# In[18]:


# Lorsque le coefficient est nul, la distribution est symétrique
# Lorsque le coefficient est négatif, la distribution est décalée vers la droite et étalée vers la gauche
# Lorsque le coefficient est positif, la distribution est décalée vers la gauche et étalée vers la droite

df.skew()


# ## Matrice des corrélations

# Etat de nos corrélations :

# In[19]:


# Nous avons selectionné toutes nos variables quantitatives. On n'utilise pas les variables qui sont catégorielles et de type 0 et 1.
df.iloc[:,0:10].corr()


# Visualisation de cette matrice pour une meilleure lecture :

# In[20]:


f,ax = plt.subplots(figsize=(8,8))

sns.heatmap(df.iloc[:,0:10].corr(), annot = True, linewidths=0.1, fmt= '.1f',ax=ax, cmap="YlGnBu")

plt.show()


# Nos "plus fortes" corrélations positives sont entre les variables :
# - Vertical_Distance_To_Hydrology et Horizontal_Distance_To_Hydrology
# - Horizontal_Distance_To_Roadways et Elevation
# - Hillshade_3pm et Aspect
# - Hillshade_3pm et Hillshade_Noon
# 
# 
# Nos "plus fortes" correlations négatives sont entre les variables :
# - Hillshade_9am et Aspect
# - Hillshade_Noon et Slope
# - Hillshade_3pm et Hillshade_9am
# 
# Puisque nous avons des corrélations négatives et positives dans ces premières colonnes, il n'y aura pas d'effet de taille pour une analyse de composantes principales.

# ## Visualisation des données

# #### Visualisation des corrélations positives

# In[21]:


list_x1 = ['Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_3pm', 'Hillshade_3pm']
list_y1 = ['Horizontal_Distance_To_Hydrology', 'Elevation','Aspect', 'Hillshade_Noon']

for i in range(0,4):
    sns.scatterplot(list_x1[i], list_y1[i], hue = "Cover_Type", palette=sns.color_palette("hls", 7), data = df)
    plt.show()


# Les corrélations entre les variables 'Aspect' et 'Hillshade_3pm', puis entre les variables 'Hillshade_Noon' et 'Hillshade_3pm' ont une forme particulière.
# 
# Quelques interprétations d'un point de vue visuel :
# - Dans le premier graphique, nous pouvons voir que pour le couvert 7 (Krummholz), ce type d'arbre ne pousse vraiment lorsqu'il est proche d'une source d'eau.
# En revanche pour les types 3 (Ponderosa Pine) et 6 (Douglas-fir), nous avons besoin d'avoir une source d'eau proche pour qu'ils poussent
# - Dans le deuxième graphique, nous pouvons distinguer plusieurs clusters selon l'élévation. Nous remarquons que peu importe la distance horizontale à la route la plus proche, seul le degré d'élévation va distinguer les différents types de couverts.

# #### Visualisation des corrélations négatives

# In[22]:


list_x2 = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
list_y2 = ['Aspect', 'Slope','Hillshade_9am']

for i in range(0,3):
    sns.scatterplot(list_x2[i], list_y2[i], hue = "Cover_Type", palette=sns.color_palette("hls", 7), data = df)
    plt.show()


# Les trois corrélations ont une forme particulière.

# #### Histogrammes de nos variables quantitatives

# In[19]:


for i in range(0,10):
    sns.distplot(df.iloc[:,i])
    plt.show()


# Nous pouvons ici insister sur l'importance de la représentation visuelle des variables quantitatives. En effet, il n'est pas pertinent de se fier seulement aux coefficients d'asymétries (skewness) calculés précédemment pour étudier l'allure des distributions. Les histogrammes de l'Elevation et de l'Aspect contiennent différents "pics", informations non identifiables avec les coefficients d'asymétries.

# #### Fréquence des zones sauvages selon les types de couverts forestiers

# In[23]:


for i in [1,2,3,4]:    
    sns.barplot(x = "Cover_Type", y = "Wilderness_Area" + str(i), data = df)
    plt.show()


# La distribution des zones sauvages par type de couverts forestiers n'est pas équilibrée. Des valeurs de types de couverts sont même manquantes la plupart du temps.

# In[24]:


for i in range(0,10):
    sns.boxplot(df.Cover_Type, df.iloc[:,i])
    plt.show()


# A travers la visualisation de ces différents boxplots (visualisation de la distribution des valeurs selon les quantiles d'un attribut) sur nos variables quantitatives différenciés par les types de couverture forestière, nous obvervons que des valeurs aberrantes sont présentes dans notre jeu de donnée quantitatif.
# Ces valeurs seront à prendre en compte dans notre modélisation.

# In[25]:


df.describe()


# ## Modélisation

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[27]:


import sklearn.datasets as sk_d
import sklearn.model_selection as sk_ms
import sklearn.ensemble as sk_e
import sklearn.svm as sk_s
import sklearn.linear_model as sk_lm 
import sklearn.preprocessing as sk_p
import sklearn.metrics as sk_m
import sklearn.decomposition as sk_de
import sklearn.cluster as sk_c
import warnings
warnings.filterwarnings('ignore')


# #### Normalisation des données

# In[28]:


scaler = sk_p.MinMaxScaler()
X_norm = scaler.fit_transform(df.iloc[:,0:10])
X_norm = pd.DataFrame(X_norm, columns = df.columns[0:10], index = df.index)
X_norm


# Nous souhaitons d'abord normaliser nos données quantitatives continues, au cas où certains modèles que nous allons utiliser seraient sensibles à la variance du dataset. Ce X_norm servira uniquement pour l'analyse en composantes principales (puisque les variables catégorielles vont biaiser l'ACP).

# In[29]:


scaler = sk_p.MinMaxScaler()
X_norm = scaler.fit_transform(df.iloc[:,0:10])
X_norm = pd.DataFrame(X_norm, columns = df.columns[0:10], index = df.index)
X_norm_1 = pd.merge(X_norm,df.iloc[:,11:],how='left',left_index=True,right_index=True)
X_norm_1


# Concaténation des données normalisées avec les données catégorielles pour la modélisation.

# #### Performances avec trois familles de modèles

# In[30]:


# Découpage des données en apprentissage/test
X_train, X_test = sk_ms.train_test_split(X_norm_1,test_size=0.2, random_state = 300)

# Découpage du test en validation/test
X_val, X_test = sk_ms.train_test_split(X_test,test_size=0.5, random_state = 300)

# Sélection de notre variable à prédire
Y_train = X_train.loc[:,'Cover_Type']
Y_val = X_val.loc[:,'Cover_Type']
Y_test = X_test.loc[:,'Cover_Type']

# Nous avons déjà notre target Y, donc on supprime la colonne que l'on cherche à prédire dans X
X_train.drop(['Cover_Type'],axis=1,inplace=True)
X_val.drop(['Cover_Type'],axis=1,inplace=True)
X_test.drop(['Cover_Type'],axis=1,inplace=True)


# In[31]:


# 3 familles de modèles :
models = {}

# 3 modèles de forêts aléatoires avec des tailles de forêts différentes
for n_estimator_i in [10,50,100]:
    models['randomForest_{}'.format(n_estimator_i)] = sk_e.RandomForestClassifier(n_estimators=n_estimator_i)
    
# 3 modèles de regression régularisée
for alpha_i in [1,100,1000]:
    models['linearRegression_{}'.format(alpha_i)] = sk_lm.RidgeClassifier(alpha=alpha_i)
    
# 3 modèles de machines à vecteurs de support (SVM) avec des paramètres de régularisation différents
for c_i in [1,50,100]:
    models['SVC_{}'.format(c_i)] = sk_s.LinearSVC(C=c_i)


# In[32]:


# Apprentissage des modèles sur le jeu d'apprentissage
val_results = []

for model_i in models.keys(): 
    models[model_i].fit(X_train,Y_train) 
    Yp_val = models[model_i].predict(X_val)
    acc = sk_m.accuracy_score(Y_val,Yp_val)
    pre = sk_m.precision_score(Y_val,Yp_val,average='macro')
    rec = sk_m.recall_score(Y_val,Yp_val,average='macro')
    val_results.append([model_i,round(acc*100,0),round(pre*100,1),round(rec*100,1)])
    
val_results = pd.DataFrame(val_results,columns=['model','acc','pre','rec'])
val_results


# Quelques définitions avant l'interprétation :
# - Accuracy : L'accuracy est le ratio du nombre de valeurs bien prédites sur la somme des valeurs que l'on cherche à prédire observées totale.
# - Precision : La précision est le nombre de résultats positifs correctement identifiés divisé par le nombre de tous les résultats positifs.
# - Recall : Le rappel est le nombre de résultats positifs correctement identifiés divisé par le nombre de tous les échantillons qui auraient dû être identifiés comme positifs.

# Pour le modèle foret aléatoire, le meilleur n_estimateur (nombre d'arbres) serait le n=100 parmis les 3 configurations(10,50,100) avec 86 en accuracy, 86.3 en precision et 86.6 en recall.
# Plus le nombre d'arbres augmente, plus le modèle est performant mais à un certain limite car si on augmente le n_estimateur au dela de 100 (par exemple 200) la precision, l'accurency et le recall diminuent.
# 
# 
# 
# Pour le modèle de régression linéaire avec un coefficient de pénalisation de 1 obtient le meilleur score, il sera pris car le modele sera plus performant.
# 
# 
# 
# Le modèle de SVC avec un terme de régularisation de 1 obtient le meilleur score.
# 
# 
# 
# Mais le meilleur parmi les trois est le RandomForest.

# #### Apprentissage des modèles sur le jeu d'apprentissage + validation

# In[33]:


models_selected = ['randomForest_100','linearRegression_1','SVC_50']

X_train_val = X_train.append(X_val,ignore_index=True)
Y_train_val = Y_train.append(Y_val,ignore_index=True)

test_results = []

for model_i in models_selected: 
    models[model_i].fit(X_train_val,Y_train_val)
    Yp_test = models[model_i].predict(X_test)
    acc = sk_m.accuracy_score(Y_test,Yp_test)
    pre = sk_m.precision_score(Y_test,Yp_test,average='macro')
    rec = sk_m.recall_score(Y_test,Yp_test,average='macro')
    test_results.append([model_i,round(acc*100,0),round(pre*100,1),round(rec*100,1)])

test_results = pd.DataFrame(test_results,columns=['model','acc','pre','rec'])
test_results


# In[34]:


def computeFmeasure(x):
    return round(2*(x['pre']*x['rec'])/(x['pre']+x['rec']),1)

test_results['F-measure'] = test_results.loc[:,['pre','rec']].apply(lambda x: computeFmeasure(x),axis=1)
test_results


# La F-Measure ou F1 dans certains cas, est une moyenne harmonique de la précision (pre) et du rappel (rec) (c'est une moyenne arithmétique inverse, des inverses de chaque termes).

# Etant donné nos résultats, le modèle le plus performant serait le Random Forest Classifier avec un paramètre de 100.
# Il est possible d'améliorer les performances des autres modèles sélectionnés.

# #### Résultats par classes à l'aide d'une matrice de confusion :

# In[35]:


from sklearn.metrics import confusion_matrix


# In[36]:


# Random Forest // Résultat de la classification par classe avec le paramètre 100 (meilleur modèle) :
model_rf = sk_e.RandomForestClassifier(n_estimators=100)
model_rf_fit = model_rf.fit(X_train_val, Y_train_val)
cf_matrix_rf = pd.DataFrame(confusion_matrix(Y_test, model_rf_fit.predict(X_test)))
cf_matrix_rf


# In[37]:


# Ridge Classifier // Résultat de la classification par classe avec le paramètre 1 (meilleur modèle) :
model_rc = sk_lm.RidgeClassifier(alpha=100)
model_rc_fit = model_rf.fit(X_train_val, Y_train_val)
cf_matrix_rc = pd.DataFrame(confusion_matrix(Y_test, model_rf_fit.predict(X_test)))
cf_matrix_rc


# In[38]:


# SVC // Résultat de la classification par classe avec le paramètre 50 (meilleur modèle) :
model_svc = sk_s.LinearSVC(C=50)
model_svc_fit = model_rf.fit(X_train_val, Y_train_val)
cf_matrix_svc = pd.DataFrame(confusion_matrix(Y_test, model_rf_fit.predict(X_test)))
cf_matrix_svc


# Des résultats assez diverses, mais une tendance semble se dégager : les erreurs de prédictions se retrouvent bien souvent aux mêmes endroits dans les différentes matrices de confusion. Par exemple, à la 6 ème ligne et 3 ème colonne (24 à 28) pour chaque matrice de confusion, ainsi qu'à la première ligne et dernière colonne (11 à 12).

# # Réduction de dimension

# #### Importance des attributs

# In[39]:


importance = [round(x*100,0) for x in models['randomForest_100'].feature_importances_]
importance = pd.DataFrame(importance,X_train.columns)
importance.rename(columns={0:'feat_importance'},inplace=True)
importance.sort_values(by='feat_importance',ascending=False)


# Pour l'importance des attributs, nous pouvons observer que la plupart des types de sols n'ont pas réellement d'impact dans notre modélisation. Nous pouvons donc les retirer et ne garder que les 2 ou 3 premiers attributs pour simplifier notre modélisation à l'avenir.

# ### Analyse en composantes principales

# In[40]:


pca = sk_de.PCA(n_components=2)
components = pca.fit_transform(X_norm)

X_red = pd.DataFrame(components,columns = ['principal component 1', 'principal component 2'],index=df.index)
X_red = pd.merge(X_red,df,how='left',left_index=True, right_index=True)


# In[41]:


# Représentation de la projection selon le Cover_Type
# Projection en 2 dimensions de notre jeu de données :

import matplotlib.pyplot as plt
import matplotlib.lines as plt_l
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
labels = X_red.Cover_Type.unique()
labels_text = {1: 'Spruce/Fir', 2: 'Lodgepole Pine', 3: 'Ponderosa Pine', 4:'Cottonwood/Willow', 5:'Aspen', 6:'Douglas-fir', 7:'Krummholz'}
colors = {1:'r',2:'g',3:'b',4:'m',5:'y',6:'k',7:'xkcd:orange'}
for label_i in labels:
    indicesToKeep = X_red['Cover_Type'] == label_i
    ax.scatter(X_red.loc[indicesToKeep, 'principal component 1'], X_red.loc[indicesToKeep, 'principal component 2']
               , c = colors[label_i], s = 50,label=labels_text[label_i])
ax.legend(labels_text)
ax.grid()


# L'analyse en composantes principales distinguée par les couvertures forestières montre:
# 
# - Les couverts Forestiers Lodgepole Pine (2), Ponderosa Pine (3) et Cottonwood/Willow (4) se situent du coté de la composante 2, ils sont corrélées avec l'axe 2 (si on trace une première bissectrice en partant de 0, la plupart des points se retrouvent au dessus de cet axe).
# - Pour le couvert Forestier Krummholz (7), une tendance semble se dégager vers l'axe 1.
# - Le couvert Forestier Krummholz Aspen (5) est centré, pas de corrélation avec les 2 axes.
# 

# In[42]:


# Représentation selon une des variables d'entrée
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
points = ax.scatter(X_red.loc[:,'principal component 1'], X_red.loc[:, 'principal component 2']
               , c = X_red.loc[:,'Elevation'], s = 50)
v1 = np.linspace(X_red.loc[:,'Elevation'].min(),X_red.loc[:,'Elevation'].max(),5, endpoint=True)
cb = fig.colorbar(points,ticks=v1)

ax.grid()


# L'analyse en composantes principales différenciés par l'élevation est intéressante, on constate une tendance qui se dégage vers l'axe 2 et donc une corrélation importante avec l'axe 2.

# In[ ]:




