"""Module d'évaluation rigoureuse, de validation croisée et d'optimisation des modèles.

Implémente :
- Validation croisée stratifiée (Stratified K-Fold, k=5).
- Métriques complètes : Accuracy, Precision, Recall, F1-score (macro & weighted), ROC-AUC.
- Génération et sauvegarde des matrices de confusion et courbes ROC comparatives.
- Optimisation d'hyperparamètres via GridSearchCV / RandomizedSearchCV.
- Analyse d'importance des variables (Feature Importance intrinsèque et permutation importance).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.data_loader import (
    DEFAULT_PROCESSED_DIR,
    load_raw_data,
    prepare_features_and_target,
    split_data,
)
from src.models import build_pipeline, get_default_models, get_hyperparameter_grids
from src.preprocessing import create_preprocessor, get_feature_names, identify_column_types
from src.utils import (
    ensure_dir,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_models_comparison,
    plot_roc_curves,
    save_json,
    save_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")


def evaluate_cross_validation(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Évalue un pipeline via Stratified K-Fold CV sur plusieurs métriques.

    Args:
        pipeline: Pipeline Scikit-learn (preprocessor + model).
        X: Variables d'entraînement.
        y: Variable cible.
        cv_folds: Nombre de plis (k=5 par défaut).
        random_state: Graine de reproductibilité.

    Returns:
        Dict[str, float]: Moyennes et écarts-types des métriques de validation.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
        "cv_precision_std": float(np.std(cv_results["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
        "cv_recall_std": float(np.std(cv_results["test_recall"])),
        "cv_f1_weighted_mean": float(np.mean(cv_results["test_f1_weighted"])),
        "cv_f1_weighted_std": float(np.std(cv_results["test_f1_weighted"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
    }
    return summary


def compute_test_metrics(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray, Tuple[np.ndarray, np.ndarray, float]]:
    """Calcule les métriques d'évaluation sur le jeu de test indépendant.

    Args:
        pipeline: Pipeline entraîné.
        X_test: Jeu de test.
        y_test: Vraies étiquettes du jeu de test.

    Returns:
        Tuple contenant :
        - dictionnaire des métriques scalaires,
        - matrice de confusion,
        - tuple ROC (fpr, tpr, roc_auc).
    """
    y_pred = pipeline.predict(X_test)
    y_prob = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(X_test)
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1_weighted": float(f1_w),
        "test_f1_macro": float(f1_m),
        "test_roc_auc": float(auc_score),
    }

    return metrics, cm, (fpr, tpr, float(auc_score))


def benchmark_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_dir: Path | str = REPORTS_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Entraîne et compare tous les modèles sur CV et Test set.

    Args:
        X_train, y_train: Jeu d'entraînement.
        X_test, y_test: Jeu de test.
        numeric_cols, categorical_cols: Types de variables.
        output_dir: Répertoire de sortie des rapports.

    Returns:
        Tuple[DataFrame comparatif, Dictionnaire des pipelines fittés, Nom du meilleur modèle].
    """
    ensure_dir(output_dir)
    ensure_dir(FIGURES_DIR)

    models_dict = get_default_models()
    records: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Any] = {}
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    logger.info(f"Démarrage du benchmark comparatif sur {len(models_dict)} algorithmes...")

    for name, raw_model in models_dict.items():
        logger.info(f"--- Évaluation de : {name} ---")
        preprocessor = create_preprocessor(numeric_cols, categorical_cols)
        pipe = build_pipeline(preprocessor, raw_model)

        # 1. Validation croisée sur le Train set
        cv_summary = evaluate_cross_validation(pipe, X_train, y_train, cv_folds=5)

        # 2. Entraînement final sur tout le Train set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # 3. Évaluation sur le Test set indépendant
        test_metrics, cm, roc_tuple = compute_test_metrics(pipe, X_test, y_test)
        roc_data[name] = roc_tuple

        # 4. Tracé et sauvegarde de la matrice de confusion
        cm_path = FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix(
            cm,
            class_names=["Échec (<10)", "Réussite (>=10)"],
            output_path=cm_path,
            title=f"Matrice de Confusion - {name}",
        )

        record = {"model": name, **cv_summary, **test_metrics}
        records.append(record)

    # Tracé comparatif des courbes ROC
    plot_roc_curves(
        roc_data,
        output_path=FIGURES_DIR / "roc_curves_comparison.png",
        title="Courbes ROC Comparatives des Modèles (Test Set)",
    )

    results_df = pd.DataFrame(records).sort_values("test_f1_weighted", ascending=False)
    summary_csv = Path(output_dir) / "benchmark_summary.csv"
    results_df.to_csv(summary_csv, index=False)
    save_json(results_df.to_dict(orient="records"), Path(output_dir) / "benchmark_metrics.json")

    # Barplot comparatif
    plot_models_comparison(
        results_df,
        metric_cols=["test_accuracy", "test_f1_weighted", "test_roc_auc"],
        output_path=FIGURES_DIR / "models_benchmark_barchart.png",
    )

    best_model_name = results_df.iloc[0]["model"]
    logger.info(f"Modèle le plus performant identifié : {best_model_name}")
    return results_df, fitted_pipelines, best_model_name


def tune_best_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    cv_folds: int = 5,
) -> Tuple[Any, Dict[str, Any]]:
    """Optimise les hyperparamètres du meilleur modèle via GridSearchCV.

    Args:
        model_name: Nom du modèle sélectionné.
        X_train, y_train: Données d'entraînement.
        X_test, y_test: Données de test.
        numeric_cols, categorical_cols: Colonnes typées.
        cv_folds: Nombre de plis de validation croisée.

    Returns:
        Tuple[Meilleur Pipeline optimisé, Dictionnaire de synthèse].
    """
    logger.info(f"Démarrage de l'optimisation GridSearchCV pour {model_name}...")
    models_dict = get_default_models()
    grids = get_hyperparameter_grids()

    if model_name not in models_dict or model_name not in grids:
        raise ValueError(f"Modèle {model_name} non supporté pour le tuning.")

    base_model = models_dict[model_name]
    param_grid = grids[model_name]

    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    pipe = build_pipeline(preprocessor, base_model)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Évaluation sur le test set
    test_metrics, cm, roc_tuple = compute_test_metrics(best_pipeline, X_test, y_test)

    # Sauvegarde du modèle optimisé
    model_save_path = ensure_dir(MODELS_DIR) / "best_model.joblib"
    save_model(best_pipeline, model_save_path)

    # Matrice de confusion du modèle optimisé
    plot_confusion_matrix(
        cm,
        class_names=["Échec (<10)", "Réussite (>=10)"],
        output_path=FIGURES_DIR / "confusion_matrix_best_tuned.png",
        title=f"Matrice de Confusion Optimisée - {model_name}",
    )

    tuning_report = {
        "best_model_name": model_name,
        "best_params": best_params,
        "best_cv_f1_weighted": float(best_cv_score),
        "test_metrics": test_metrics,
        "saved_path": str(model_save_path),
    }
    save_json(tuning_report, REPORTS_DIR / "tuning_report.json")
    logger.info(f"Modèle optimisé sauvegardé dans {model_save_path}")

    return best_pipeline, tuning_report


def analyze_feature_importance(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    """Analyse l'importance des variables (intrinsèque et par permutation).

    Args:
        pipeline: Pipeline entraîné (contenant préprocesseur et classifieur).
        X_train: Données d'entraînement pour extraire les noms des features.
        X_test: Données de test pour la permutation importance.
        y_test: Cibles de test.
        top_n: Nombre de features prioritaires à afficher et sauvegarder.

    Returns:
        pd.DataFrame: Tableau récapitulatif des importances.
    """
    logger.info("Calcul de l'importance des variables...")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor)

    # 1. Feature Importance intrinsèque (arbres / forêts / coefficients logistiques)
    importances: Optional[np.ndarray] = None
    method_name = "Permutation Importance"

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        method_name = "Tree Feature Importance (Gini)"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        method_name = "Logistic Regression (|Coefficients|)"

    if importances is not None and len(importances) == len(feature_names):
        df_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        plot_feature_importances(
            feature_names=df_importance["feature"],
            importances=df_importance["importance"],
            top_n=top_n,
            output_path=FIGURES_DIR / "feature_importance_intrinsic.png",
            title=f"Top {top_n} Variables - {method_name}",
        )
    else:
        df_importance = pd.DataFrame()

    # 2. Permutation Importance (méthode agnostique au modèle sur le jeu de test)
    perm_result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="f1_weighted"
    )

    df_perm = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    df_perm.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_feature_importances(
        feature_names=df_perm["feature"],
        importances=df_perm["importance_mean"],
        top_n=top_n,
        output_path=FIGURES_DIR / "feature_importance_permutation.png",
        title=f"Permutation Importance (Impact sur F1-Score Test)",
    )

    return df_perm


def main() -> None:
    """Point d'entrée CLI pour lancer le benchmark complet et le tuning."""
    parser = argparse.ArgumentParser(description="Évaluation, comparaison et optimisation des modèles.")
    parser.add_argument(
        "--include-grades",
        action="store_true",
        help="Inclure les notes intermédiaires G1 et G2 dans la modélisation.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=True,
        help="Optimise les hyperparamètres du meilleur modèle après comparaison.",
    )
    args = parser.parse_args()

    # Chargement et préparation des données
    df = load_raw_data()
    X, y = prepare_features_and_target(df, include_intermediate_grades=args.include_grades)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    num_cols, cat_cols = identify_column_types(X_train)

    # 1. Benchmark complet de tous les modèles
    benchmark_df, pipelines, best_model_name = benchmark_all_models(
        X_train, y_train, X_test, y_test, num_cols, cat_cols
    )

    print("\n================== RÉSULTATS DU BENCHMARK ==================")
    print(
        benchmark_df[
            [
                "model",
                "cv_accuracy_mean",
                "cv_f1_weighted_mean",
                "test_accuracy",
                "test_f1_weighted",
                "test_roc_auc",
            ]
        ].to_string(index=False)
    )

    # 2. Hyperparameter Tuning du modèle gagnant
    if args.tune:
        best_pipe, tuning_report = tune_best_model(
            best_model_name, X_train, y_train, X_test, y_test, num_cols, cat_cols
        )
        print("\n================== RAPPORT DE TUNING ==================")
        print(f"Modèle : {best_model_name}")
        print(f"Meilleurs hyperparamètres : {tuning_report['best_params']}")
        print(f"Test F1-Weighted : {tuning_report['test_metrics']['test_f1_weighted']:.4f}")
        print(f"Test ROC-AUC     : {tuning_report['test_metrics']['test_roc_auc']:.4f}")

        # 3. Analyse de l'importance des variables
        analyze_feature_importance(best_pipe, X_train, X_test, y_test)


if __name__ == "__main__":
    main()
