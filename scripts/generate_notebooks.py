"""Script pour générer les notebooks pédagogiques et interactifs du projet.

Crée :
- notebooks/01_eda.ipynb : Analyse exploratoire approfondie, distributions, corrélations.
- notebooks/02_pipeline.ipynb : Expérimentations, comparaison d'algorithmes et optimisation.
"""

from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_eda_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    # Titre et présentation
    cells.append(
        nbf.v4.new_markdown_cell(
            """# 📊 Notebook 01 : Analyse Exploratoire des Données (EDA)
## Projet : Student Performance Prediction
### Objectif académique :
Explorer le jeu de données *Student Performance* (UCI Machine Learning Repository), inspecter la variable cible, identifier les corrélations significatives, détecter les anomalies et comprendre l'impact des caractéristiques socio-démographiques et habitudes de vie sur la réussite scolaire."""
        )
    )

    # Imports
    cells.append(
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

# Ajout de la racine du projet au path
sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, prepare_features_and_target
from src.utils import set_plot_style

set_plot_style()
print("Bibliothèques chargées et style configuré avec succès.")"""
        )
    )

    # Chargement
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Chargement et vue d'ensemble du jeu de données brut
Le fichier `student-mat.csv` regroupe les performances en mathématiques des élèves des lycées Gabriel Pereira (GP) et Mousinho da Silveira (MS)."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """df_raw = load_raw_data("../data/raw/student-mat.csv")
print(f"Dimensions : {df_raw.shape[0]} étudiants x {df_raw.shape[1]} variables")
df_raw.head()"""
        )
    )

    # Types de données et intégrité
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Intégrité des données : types, valeurs manquantes et doublons"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """print("--- Valeurs manquantes ---")
missing_count = df_raw.isnull().sum().sum()
print(f"Total des valeurs manquantes : {missing_count}")

print("\\n--- Doublons stricts ---")
duplicates_count = df_raw.duplicated().sum()
print(f"Nombre de doublons : {duplicates_count}")

df_raw.info()"""
        )
    )

    # Distribution de la note finale et de la cible binaire
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Analyse de la variable cible : Notes finales (G3) et validation (>= 10)
En France et au Portugal, la validation académique s'obtient à partir de 10/20.
Analysons la distribution de `G3` ainsi que le taux de réussite binaire associé."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Distribution des notes G3
sns.histplot(df_raw['G3'], bins=20, kde=True, color='royalblue', ax=ax1)
ax1.axvline(10, color='red', linestyle='--', label='Seuil de validation (10/20)')
ax1.set_title("Distribution de la note finale G3")
ax1.set_xlabel("Note G3 (sur 20)")
ax1.set_ylabel("Nombre d'étudiants")
ax1.legend()

# Distribution binaire du statut de réussite
df_raw['passed'] = (df_raw['G3'] >= 10).map({True: 'Réussi (>=10)', False: 'Échec (<10)'})
passed_counts = df_raw['passed'].value_counts()
ax2.pie(passed_counts, labels=passed_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140)
ax2.set_title("Répartition Réussite vs Échec")

plt.show()"""
        )
    )

    # Analyse des facteurs sociologiques et habitudes d'études
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Analyse des habitudes d'étude et antécédents scolaires
Facteurs clés à analyser :
- **Temps d'étude hebdomadaire** (`studytime` : 1 = <2h, 2 = 2-5h, 3 = 5-10h, 4 = >10h)
- **Échecs passés** (`failures` : 0 à 4)
- **Absences scolaires** (`absences`)
- **Soutien scolaire supplémentaire** (`schoolsup`)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temps d'étude vs Note G3
sns.boxplot(x='studytime', y='G3', data=df_raw, palette='Blues', ax=axes[0, 0])
axes[0, 0].set_title("Note G3 selon le temps d'étude hebdomadaire")
axes[0, 0].set_xlabel("Temps d'étude (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
axes[0, 0].set_ylabel("Note G3")

# Échecs scolaires passés vs G3
sns.boxplot(x='failures', y='G3', data=df_raw, palette='Reds', ax=axes[0, 1])
axes[0, 1].set_title("Note G3 selon le nombre d'échecs passés")
axes[0, 1].set_xlabel("Nombre d'échecs passés")
axes[0, 1].set_ylabel("Note G3")

# Taux de réussite selon le soutien scolaire (schoolsup)
sns.countplot(x='schoolsup', hue='passed', data=df_raw, palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title("Réussite selon l'aide scolaire (schoolsup)")
axes[1, 0].set_xlabel("Bénéficie d'aide scolaire")
axes[1, 0].set_ylabel("Effectif")

# Absences vs G3 avec distinction réussite/échec
sns.scatterplot(x='absences', y='G3', hue='passed', alpha=0.7, data=df_raw, palette={'Réussi (>=10)': '#2ecc71', 'Échec (<10)': '#e74c3c'}, ax=axes[1, 1])
axes[1, 1].set_title("Absences vs Note G3")
axes[1, 1].set_xlabel("Nombre d'absences")
axes[1, 1].set_ylabel("Note G3")

plt.tight_layout()
plt.show()"""
        )
    )

    # Analyse de l'impact de la vie personnelle et de la famille
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Habitudes de vie, cadre familial et consommation d'alcool
- `Dalc` (Workday alcohol consumption, 1-5) & `Walc` (Weekend alcohol consumption, 1-5)
- `Medu` & `Fedu` : Niveau d'instruction des parents (0 à 4)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Alcool le weekend vs Note G3
sns.boxplot(x='Walc', y='G3', data=df_raw, palette='Oranges', ax=ax1)
ax1.set_title("Impact de la consommation d'alcool le weekend sur G3")
ax1.set_xlabel("Consommation d'alcool le weekend (1=très faible, 5=très élevée)")
ax1.set_ylabel("Note G3")

# Éducation de la mère vs G3
sns.boxplot(x='Medu', y='G3', data=df_raw, palette='Purples', ax=ax2)
ax2.set_title("Impact du niveau d'études de la mère sur G3")
ax2.set_xlabel("Éducation de la mère (0=aucune, 4=enseignement supérieur)")
ax2.set_ylabel("Note G3")

plt.show()"""
        )
    )

    # Matrice de corrélation
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Matrice de corrélation des variables quantitatives"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """numeric_df = df_raw.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title("Matrice de Corrélation Linéaire de Pearson")
plt.show()"""
        )
    )

    # Synthèse de l'EDA
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. Synthèse des constats de l'EDA
1. **Déséquilibre de classe modéré** : ~67% de réussite contre ~33% d'échec. Un découpage stratifié (*Stratified Train/Test Split*) et des métriques pondérées ($F1_{weighted}$, $ROC\\text{-}AUC$) sont indispensables.
2. **Facteurs négatifs les plus influents** : L'historique des échecs antérieurs (`failures`) est le prédicteur négatif le plus fort, suivi d'une forte consommation d'alcool le weekend (`Walc`).
3. **Facteurs positifs** : Le niveau d'études des parents (`Medu`, `Fedu`), l'accès à internet et le temps d'étude hebdomadaire (`studytime`) favorisent significativement les résultats.
4. **Intégrité du dataset** : Aucune valeur manquante détectée dans les données brutes, mais une imputation défensive (`SimpleImputer`) doit être maintenue dans les pipelines Scikit-Learn pour la robustesse en production."""
        )
    )

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "01_eda.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook 01_eda.ipynb généré.")


def build_pipeline_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    # Titre
    cells.append(
        nbf.v4.new_markdown_cell(
            """# ⚙️ Notebook 02 : Prototypage de Pipeline, Modélisation et Optimisation
## Projet : Student Performance Prediction
### Objectif :
1. Mettre en place un pipeline Scikit-Learn étanche (aucun *data leakage*).
2. Valider et comparer 5 algorithmes sous validation croisée stratifiée ($k=5$).
3. Évaluer les performances sur jeu de test indépendant (ROC-AUC, Confusion Matrix, F1-Score).
4. Optimiser les hyperparamètres via `GridSearchCV`.
5. Interpréter l'importance des variables (*Permutation Importance*)."""
        )
    )

    # Imports
    cells.append(
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, prepare_features_and_target, split_data
from src.preprocessing import create_preprocessor, identify_column_types
from src.models import get_default_models, get_hyperparameter_grids, build_pipeline
from src.evaluate import benchmark_all_models, tune_best_model, analyze_feature_importance
from src.utils import set_plot_style

set_plot_style()
print("Modules du projet importés avec succès.")"""
        )
    )

    # Chargement et partitionnement
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Chargement et découpage Train / Test sans fuite d'information
Pour évaluer l'impact socio-démographique pur, nous excluons les notes intermédiaires `G1` et `G2`."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """df = load_raw_data("../data/raw/student-mat.csv")
X, y = prepare_features_and_target(df, target_type="binary", include_intermediate_grades=False)

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42, stratify=True)
num_cols, cat_cols = identify_column_types(X_train)

print(f"Échantillon Train : {X_train.shape[0]} étudiants")
print(f"Échantillon Test  : {X_test.shape[0]} étudiants")
print(f"Variables numériques   : {len(num_cols)}")
print(f"Variables catégorielles : {len(cat_cols)}")"""
        )
    )

    # Architecture du ColumnTransformer
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Construction du ColumnTransformer Scikit-Learn
- **Variables numériques** : Imputation médiane + `StandardScaler`
- **Variables catégorielles** : Imputation mode + `OneHotEncoder(handle_unknown='ignore')`"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """preprocessor = create_preprocessor(num_cols, cat_cols)
preprocessor"""
        )
    )

    # Benchmark complet
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Comparaison systématique des modèles (Stratified K-Fold k=5)
Algorithmes testés :
1. **Logistic Regression** (baseline linéaire pondérée)
2. **K-Nearest Neighbors** (approche par proximité)
3. **Decision Tree Classifier** (arbre cartographique)
4. **Random Forest Classifier** (forêt aléatoire d'arbres)
5. **Gradient Boosting Classifier** (ensemble séquentiel avec boosting)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """benchmark_df, fitted_pipelines, best_name = benchmark_all_models(
    X_train, y_train, X_test, y_test, num_cols, cat_cols, output_dir="../reports"
)

display_cols = [
    "model", "cv_accuracy_mean", "cv_f1_weighted_mean", 
    "test_accuracy", "test_f1_weighted", "test_roc_auc"
]
benchmark_df[display_cols]"""
        )
    )

    # Visualisation des métriques
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Visualisation comparative des performances"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, ax = plt.subplots(figsize=(10, 5))
melted = pd.melt(
    benchmark_df, 
    id_vars=['model'], 
    value_vars=['test_accuracy', 'test_f1_weighted', 'test_roc_auc'], 
    var_name='Métrique', 
    value_name='Score'
)
sns.barplot(data=melted, x='model', y='Score', hue='Métrique', ax=ax, palette='mako')
ax.set_ylim([0, 1.05])
ax.set_title("Comparaison des Modèles sur le Test Set")
plt.xticks(rotation=15)
plt.show()"""
        )
    )

    # Hyperparameter tuning
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 5. Optimisation fine des hyperparamètres (GridSearchCV) sur le meilleur modèle"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """best_pipe, tuning_report = tune_best_model(
    best_name, X_train, y_train, X_test, y_test, num_cols, cat_cols, cv_folds=5
)

print(f"Meilleurs hyperparamètres trouvés :\\n{tuning_report['best_params']}")
print(f"Score F1-Weighted en validation croisée : {tuning_report['best_cv_f1_weighted']:.4f}")
print(f"Score F1-Weighted sur Test Set           : {tuning_report['test_metrics']['test_f1_weighted']:.4f}")
print(f"ROC-AUC sur Test Set                     : {tuning_report['test_metrics']['test_roc_auc']:.4f}")"""
        )
    )

    # Importance des variables
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 6. Analyse d'importance des variables (Permutation Importance)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """perm_df = analyze_feature_importance(best_pipe, X_train, X_test, y_test, top_n=12)
perm_df.head(12)"""
        )
    )

    # Conclusion
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 7. Conclusions et Perspectives
- Le modèle optimisé et les graphiques d'évaluation sont sauvegardés dans les dossiers `models/` et `reports/figures/`.
- La démarche garantit l'absence de fuite de données et respecte les critères d'ingénierie logicielle pour un déploiement fiable."""
        )
    )

    nb.cells = cells
    with open(NOTEBOOKS_DIR / "02_pipeline.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook 02_pipeline.ipynb généré.")


if __name__ == "__main__":
    build_eda_notebook()
    build_pipeline_notebook()
