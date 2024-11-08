# ml-work
ML Work for energy prediction

Requirements: 
    numpy version: 1.26.4
    pandas version: 2.2.2
    optuna version: 4.0.0
    scikit-learn version: 1.5.1
    seaborn version: 0.13.2
    mlflow version: 1.2.0
    kaggle version: 0.2.7
    os version: posix


Estimation du Prix de l'Électricité en Espagne
Ce projet utilise un modèle de machine learning pour estimer le prix de l’électricité en Espagne en fonction de l’offre et de la demande. Le modèle prend en compte plusieurs caractéristiques (features) et est optimisé pour fournir des prédictions précises du prix en utilisant des données historiques.

Table des Matières
Contexte
Objectif
Données
Pré-requis
Structure du Code
Utilisation
Optimisation Hyperparamétrique
Évaluation du Modèle
Conclusion


Contexte
L’électricité est une ressource cruciale, et sa demande et son offre peuvent varier en fonction 
de nombreux facteurs, tels que les conditions météorologiques, les jours de la semaine, les saisons, 
et d'autres variables socio-économiques. Ce projet utilise un modèle d’apprentissage automatique (HistGradientBoostingRegressor) pour évaluer la demande d’électricité et fournir une estimation du prix.

Objectif
L'objectif principal de ce projet est de développer un modèle prédictif c
apable d’estimer le prix réel de l'électricité en Espagne en fonction de diverses caractéristiques. 
Ce modèle est optimisé pour fournir des prévisions fiables en utilisant des techniques avancées 
de machine learning et d’optimisation des hyperparamètres (Optuna).

Dataset description: 

baseline : 
Mean Price Per KW/h Baseline Pred: 57.912637711864406
-------------------------------------------------------------------
RMSE: 14.173160623162811
RMSE:  7.61
R2:  0.76
mape:  0.11

Resultats finale:
Mean Price Per KW/h Pred: 57.83611369000541
-------------------------------------------------------------------
RMSE: 4.932829257705223

RMSE:  4.93
R2:  0.99
mape:  0.07


Structure du Code
Pré-traitement des données:

Séparation des colonnes numériques et catégorielles.
Utilisation de l'imputation des valeurs manquantes et de l'encodage pour les variables catégorielles.
Pipeline de Modélisation:

Création d'un pipeline utilisant HistGradientBoostingRegressor pour le modèle de régression.
Le pipeline intègre également le préprocesseur pour traiter les données avant qu’elles soient envoyées au modèle.
Optimisation des Hyperparamètres:

Utilisation d’Optuna pour effectuer une recherche des hyperparamètres du modèle 
(tels que learning_rate, max_iter, max_leaf_nodes) afin de maximiser la performance.

Évaluation des Performances:

Mesure des performances avec les métriques suivantes : 
    RMSE (Root Mean Squared Error), 
    R2 (coefficient de détermination), 
    MAPE (Mean Absolute Percentage Error).
    Cross validation

Optimisation Hyperparamétrique
Le projet utilise Optuna pour rechercher les meilleurs hyperparamètres du modèle HistGradientBoostingRegressor. 
Optuna effectue une optimisation bayésienne sur des essais pour maximiser les performances du modèle. 
Les hyperparamètres optimisés sont :

learning_rate: Taux d'apprentissage du modèle.
max_iter: Nombre d'itérations pour l’entraînement.
max_leaf_nodes: Nombre maximal de nœuds dans les arbres.
