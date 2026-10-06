# 🎓 Student Performance Prediction — Projet Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-Passing-brightgreen?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![PEP 8](https://img.shields.io/badge/Code%20Style-PEP%208-blue)](https://pep8.org/)

Projet académique de **Machine Learning supervisé** (niveau école d'ingénieur) modélisant et prédisant la réussite scolaire des étudiants à partir de leurs caractéristiques socio-démographiques, de leur environnement familial et de leurs habitudes d'étude.

---

## 📌 1. Contexte et Problématique

- **Dataset de référence** : *Student Performance Dataset* ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance) / Cortez & Silva, 2008).
- **Problématique ML** : Classification supervisée de la réussite finale ($G3 \ge 10/20$).
- **Enjeu méthodologique & Prévention de la fuite de données (*Data Leakage*)** :
  - Dans la littérature, l'inclusion des notes intermédiaires ($G1$ et $G2$) produit une prédiction quasi-triviale en masquant les facteurs socio-éducatifs.
  - Le pipeline implémente par défaut une modélisation **exempte de $G1$ et $G2$** afin d'évaluer la réelle capacité prédictive des déterminants sociaux, familiaux et comportementaux (avec option d'inclusion pour comparaison).
  - L'ensemble du pré-traitement (imputation médiane/mode, mise à l'échelle standard, encodage one-hot) est rigoureusement encapsulé dans un `ColumnTransformer` ajusté **exclusivement** sur le sous-ensemble d'entraînement (*Train set*).

---

## 🏛️ 2. Architecture du Projet

```text
Project_Machine_Learning/
├── data/
│   ├── raw/                  # Jeu de données brut (student-mat.csv téléchargé depuis l'UCI)
│   └── processed/            # Partitions nettoyées (X_train, X_test, y_train, y_test)
├── notebooks/
│   ├── 01_eda.ipynb          # Analyse exploratoire approfondie et visualisations statistiques
│   └── 02_pipeline.ipynb     # Prototypage de pipeline, benchmark et réglage d'hyperparamètres
├── src/
│   ├── __init__.py           # Déclaration de package Python
│   ├── data_loader.py        # Téléchargement auto, contrôle de schéma et partitionnement stratifié
│   ├── preprocessing.py      # ColumnTransformer réutilisable et étanche (zéro fuite)
│   ├── models.py             # Définition des 5 classifieurs et grilles d'hyperparamètres
│   ├── evaluate.py           # Stratified K-Fold CV, métriques, courbes ROC, confusion et tuning
│   └── utils.py              # Fonctions utilitaires d'affichage graphique, sauvegarde et export
├── models/                   # Modèles optimisés entraînés (.joblib)
├── reports/                  # Rapports d'évaluation, métriques et figures exportées
│   ├── figures/              # Matrices de confusion, courbes ROC, histogrammes comparatifs
│   ├── benchmark_summary.csv # Tableau comparatif des algorithmes
│   ├── benchmark_metrics.json# Synthèse JSON des performances
│   └── tuning_report.json    # Paramètres optimaux issus du GridSearchCV
├── tests/
│   └── test_pipeline.py      # Suite de tests unitaires et d'intégration automatisés
├── conftest.py               # Configuration de l'environnement de test pytest
├── requirements.txt          # Dépendances logicielles verrouillées
└── README.md                 # Documentation complète du projet
```

---

## ⚙️ 3. Installation et Configuration

### Prérequis
- Système d'exploitation : macOS / Linux / Windows
- Python : **3.10 ou supérieur** (Python 3.11 recommandé)

### Initialisation de l'environnement virtuel

```bash
# 1. Cloner ou naviguer dans le répertoire du projet
cd Project_Machine_Learning

# 2. Création de l'environnement virtuel
python3 -m venv .venv

# 3. Activation de l'environnement virtuel
# Sur macOS / Linux :
source .venv/bin/activate
# Sur Windows :
# .venv\Scripts\activate

# 4. Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 4. Guide d'Exécution

Le projet est conçu pour une exécution flexible, soit via des **scripts en ligne de commande (CLI)** avec arguments, soit via les **Notebooks Jupyter** interactifs.

### 4.1. Chargement et préparation des données
Télécharge automatiquement le dataset officiel de l'UCI si absent, applique la séparation stratifiée et sauvegarde les partitions :

```bash
python -m src.data_loader --download --split
```
*Options disponibles :*
- `--target-type [binary|multiclass|regression]` : Choisit le type de cible (défaut : `binary`).
- `--include-grades` : Ajoute $G1$ et $G2$ dans les variables d'apprentissage.

### 4.2. Test du pipeline de prétraitement
Vérifie la transformation sans fuite et affiche les caractéristiques générées :

```bash
python -m src.preprocessing
```

### 4.3. Benchmark complet, optimisation et exports
Exécute la validation croisée stratifiée ($k=5$) sur les 5 modèles, évalue sur le jeu de test indépendant, lance l'optimisation par `GridSearchCV`, calcule l'importance des variables par permutation et exporte tous les graphiques :

```bash
python -m src.evaluate --tune
```

### 4.4. Exécution de la suite de tests automatisés

```bash
pytest tests/ -v
```

### 4.5. Lancement des Notebooks Jupyter

```bash
jupyter notebook notebooks/
```
- `01_eda.ipynb` : Analyse exploratoire pas-à-pas (distributions univariées, corrélations, boxplots socio-éducatifs).
- `02_pipeline.ipynb` : Démonstration interactive de bout en bout du pipeline ML.

---

## 📊 5. Synthèse des Résultats et Analyse Comparative

### 5.1. Comparaison des Algorithmes (Évaluation sur Test Set Indépendant, $N=79$)

| Algorithme | CV Accuracy (k=5) | CV F1-Weighted | Test Accuracy | Test F1-Weighted | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | 0.6837 | 0.6516 | **0.6835** | **0.6560** | **0.6386** |
| **Decision Tree** | 0.6016 | 0.6003 | 0.6076 | 0.6183 | 0.6299 |
| **Logistic Regression (Baseline)** | **0.6838** | **0.6876** | 0.5949 | 0.6015 | 0.5951 |
| **Gradient Boosting** | 0.6774 | 0.6589 | 0.5823 | 0.5695 | 0.4964 |
| **K-Nearest Neighbors** | 0.6836 | 0.6433 | 0.5823 | 0.5549 | 0.5221 |

> [!NOTE]
> Les résultats ci-dessus sont obtenus en **prédisant la note finale sans utiliser les notes intermédiaires G1 et G2**, ce qui reflète la difficulté réelle de prédire le parcours académique uniquement par le prisme socio-comportemental.

### 5.2. Optimisation d'Hyperparamètres (`GridSearchCV`)
Sur le modèle sélectionné (**Random Forest Classifier**), un réglage fin sur 324 combinaisons (1620 ajustements) identifie les hyperparamètres optimaux :
- `n_estimators`: 100
- `max_depth`: 12
- `min_samples_split`: 2
- `min_samples_leaf`: 2
- `max_features`: `None` (considération de toutes les dimensions à chaque partition)

Le modèle final ajusté est persigné dans `models/best_model.joblib`.

### 5.3. Facteurs Clés de la Réussite Académique (*Permutation Importance*)
L'analyse d'importance met en lumière les déterminants majeurs suivants :
1. **Échecs scolaires antérieurs (`failures`)** : Prédicteur critique négatif prépondérant.
2. **Absences (`absences`)** : Indicateur à double tranchant (les absences modérées traduisent souvent des démarches d'études personnelles alors que les absences massives précipitent l'échec).
3. **Temps d'étude hebdomadaire (`studytime`) & Soutien scolaire (`schoolsup`)** : Forte incidence sur la probabilité de franchir le seuil des 10/20.
4. **Niveau d'éducation de la mère (`Medu`)** : Facteur socio-culturel favorable statistiquement significatif.
5. **Consommation d'alcool le week-end (`Walc`)** : Facteur de risque modéré à élevé.

---

## 🛡️ 6. Standards de Qualité Logicielle

- **Typage statique (Type Hints)** : Présence systématique des annotations de type (PEP 484) pour toutes les fonctions et classes.
- **Conformité PEP 8** : Code structuré, modulaire, commenté et conforme aux conventions Python.
- **Architecture de pipeline étanche** : Aucune transformation statistique (calcul de moyenne, médiane, variance) n'est calculée sur l'ensemble de test avant le test final.
- **Reproductibilité** : Fixation stricte de la graine pseudo-aléatoire (`random_state=42`) sur l'ensemble des modules.
