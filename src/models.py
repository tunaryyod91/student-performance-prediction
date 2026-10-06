"""Module de définition des algorithmes de Machine Learning et grilles d'hyperparamètres.

Implémente et configure les 4 algorithmes fondamentaux requis :
1. Baseline : Logistic Regression
2. K-Nearest Neighbors (KNN Classifier)
3. Decision Tree Classifier
4. Random Forest Classifier (avec Gradient Boosting en complément)
Fournit également les pipelines intégrés et les grilles pour GridSearchCV / RandomizedSearchCV.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """Retourne les estimateurs de base configurés pour la classification.

    Args:
        random_state: Graine pseudo-aléatoire pour la reproductibilité.

    Returns:
        Dict[str, Any]: Dictionnaire {nom_modèle: instance_estimateur}.
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        ),
    }
    return models


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """Définit les espaces d'hyperparamètres pour l'optimisation via Grid/Randomized Search.

    Les préfixes 'classifier__' permettent de cibler directement l'estimateur
    au sein d'un Pipeline Scikit-learn complet.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionnaire de grilles d'exploration.
    """
    grids: Dict[str, Dict[str, Any]] = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "classifier__max_depth": [3, 5, 8, 12, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [5, 8, 12, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 4, 6],
            "classifier__subsample": [0.8, 1.0],
        },
    }
    return grids


def build_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    """Assemble un préprocesseur et un estimateur dans un pipeline Scikit-learn unifié.

    Args:
        preprocessor: ColumnTransformer configuré.
        model: Estimateur Scikit-learn (classifieur).

    Returns:
        Pipeline: Pipeline complet réutilisable pour fit/predict.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
