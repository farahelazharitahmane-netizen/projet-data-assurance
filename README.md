# Projet Data – DU BigData 2025-2026

## Participants

* Farah El Azhari Tahmane
* Doâa Bouasse
* Kadidiatou Mohamed Hamil Maiga
* Sonia Djounfoune


## Organisation du depot

Ce depot est structure de la maniere suivante :

* `Traitement de donnees/`
   * `Traitement Donnees Nettoyage.ipynb` : Notebook dedie a l'importation, au nettoyage des bases d'assurance, a la gestion des anomalies et a la sauvegarde des donnees nettoyees.

* `Econometrie/`
   * `Econometrie.ipynb` : Notebook contenant l'analyse descriptive (univariee, bivariee), l'ACP, la classification K-Means, et les modelisations parametriques (regression lineaire, GLM logistique, GLM Poisson).

* `Machine Learning/`
   * `Machine Learning.ipynb` : Notebook regroupant l'entrainement, l'optimisation (GridSearchCV, validation croisee) et la comparaison des performances de plusieurs modeles de Machine Learning (arbre de decision, random forest, gradient boosting, SVM).

* `Mise en production/`
   * `app.py` : Code source de l'application interactive developpee avec la bibliotheque Streamlit presentant la DataViz, les resultats d'econometrie, de machine learning et un simulateur de prediction.

* `Data/`
   * Contient les fichiers de donnees brutes (`dossier.csv`, `ressources.csv`, `temps.csv`) et les fichiers nettoyes (`dossiers_clean.csv`, `temps_clean.csv`, `ressources_clean.csv`).