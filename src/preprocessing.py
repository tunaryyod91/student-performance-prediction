"""Module de prétraitement des données et construction des pipelines Scikit-learn.

Définit un ColumnTransformer réutilisable et étanche aux fuites de données (data leakage) :
- Variables numériques : imputation par la médiane + mise à l'échelle (StandardScaler).
- Variables catégorielles : imputation par le mode + encodage one-hot (OneHotEncoder).
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_raw_data, prepare_features_and_target, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifie automatiquement les colonnes numériques et catégorielles.

    Args:
        df: DataFrame contenant les variables explicatives X.

    Returns:
        Tuple[List[str], List[str]]: (colonnes_numériques, colonnes_catégorielles).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    logger.debug(f"Détection de {len(numeric_cols)} numériques et {len(categorical_cols)} catégorielles.")
    return numeric_cols, categorical_cols


def create_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Construit un ColumnTransformer Scikit-learn pour le prétraitement des données.

    Pipeline numérique :
      - SimpleImputer(strategy='median')
      - StandardScaler() (optionnel via scale_numeric)

    Pipeline catégoriel :
      - SimpleImputer(strategy='most_frequent')
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Args:
        numeric_features: Liste explicite des noms de variables numériques.
        categorical_features: Liste explicite des noms de variables catégorielles.
        scale_numeric: Si True, applique StandardScaler aux variables numériques.

    Returns:
        ColumnTransformer: Transformateur prêt à être intégré dans un Pipeline.
    """
    # Sous-pipeline numérique
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Sous-pipeline catégoriel
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Récupère la liste des noms de colonnes résultant de la transformation One-Hot.

    Args:
        preprocessor: ColumnTransformer préalablement ajusté (fit).

    Returns:
        List[str]: Noms des variables transformées.
    """
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            # Obtenir le dernier transformateur de la chaîne
            last_step = transformer.named_steps[list(transformer.named_steps.keys())[-1]]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols).tolist()
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)
    return feature_names


def fit_and_transform_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """Ajuste le transformateur sur le Train uniquement et transforme Train & Test.

    Garantit l'étanchéité statistique (aucun data leakage du Test vers le Train).

    Args:
        X_train: Données d'entraînement.
        X_test: Données de test.
        numeric_features: Liste des colonnes numériques.
        categorical_features: Liste des colonnes catégorielles.

    Returns:
        Tuple[X_train_trans, X_test_trans, preprocessor, feature_names]
    """
    if numeric_features is None or categorical_features is None:
        num_cols, cat_cols = identify_column_types(X_train)
        numeric_features = numeric_features or num_cols
        categorical_features = categorical_features or cat_cols

    preprocessor = create_preprocessor(numeric_features, categorical_features)

    # STRICTEMENT fit sur train uniquement
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    logger.info(
        f"Prétraitement achevé :\n"
        f"  - Dimensions Train transformé : {X_train_trans.shape}\n"
        f"  - Dimensions Test transformé  : {X_test_trans.shape}\n"
        f"  - Nombre total de features    : {len(feature_names)}"
    )

    return X_train_trans, X_test_trans, preprocessor, feature_names


def main() -> None:
    """Test en ligne de commande du pipeline de prétraitement."""
    parser = argparse.ArgumentParser(description="Test du pipeline de prétraitement Scikit-learn.")
    parser.add_argument("--include-grades", action="store_true", help="Inclure G1 et G2.")
    args = parser.parse_args()

    df = load_raw_data()
    X, y = prepare_features_and_target(df, include_intermediate_grades=args.include_grades)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_t, X_test_t, preprocessor, names = fit_and_transform_splits(X_train, X_test)
    print("\n=== PIPELINE DE PRÉTRAITEMENT VALIDÉ ===")
    print(f"Features en entrée : {X_train.shape[1]}")
    print(f"Features après One-Hot Encoding : {X_train_t.shape[1]}")
    print(f"Exemples de variables encodées (10 premières) : {names[:10]}")


if __name__ == "__main__":
    main()
