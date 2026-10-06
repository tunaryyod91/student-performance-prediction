"""Module utilitaire pour le projet Student Performance Prediction.

Fournit des fonctions pour la gestion des répertoires, l'exportation
des métriques, la sauvegarde/chargement des modèles et la génération de tracés graphiques.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def ensure_dir(path: Path | str) -> Path:
    """Crée le répertoire s'il n'existe pas et renvoie un objet Path.

    Args:
        path: Chemin du répertoire à créer.

    Returns:
        Path: Objet Path pointant vers le répertoire.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Dict[str, Any], filepath: Path | str, indent: int = 4) -> None:
    """Sauvegarde un dictionnaire au format JSON.

    Args:
        data: Dictionnaire de données à sauvegarder.
        filepath: Chemin de destination.
        indent: Indentation JSON (par défaut 4).
    """
    path = Path(filepath)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Path | str) -> Dict[str, Any]:
    """Charge un fichier JSON.

    Args:
        filepath: Chemin vers le fichier JSON.

    Returns:
        Dict[str, Any]: Données chargées.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model: Any, filepath: Path | str) -> None:
    """Sauvegarde un estimateur ou pipeline Scikit-learn via joblib.

    Args:
        model: Modèle ou pipeline à persister.
        filepath: Chemin de destination (.joblib).
    """
    path = Path(filepath)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(filepath: Path | str) -> Any:
    """Charge un estimateur ou pipeline sauvegardé avec joblib.

    Args:
        filepath: Chemin du fichier .joblib.

    Returns:
        Any: Modèle restauré.
    """
    return joblib.load(filepath)


def set_plot_style() -> None:
    """Configure un style visuel soigné et reproductible pour Seaborn et Matplotlib."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Optional[Path | str] = None,
    title: str = "Matrice de Confusion",
    normalize: bool = False,
    cmap: str = "Blues",
) -> plt.Figure:
    """Trace et sauvegarde une matrice de confusion avec annotations lisibles.

    Args:
        confusion_matrix: Matrice 2D (NxN).
        class_names: Noms des classes.
        output_path: Chemin où exporter la figure si spécifié.
        title: Titre du graphique.
        normalize: Si True, affiche les pourcentages par ligne.
        cmap: Nom de la colormap Seaborn.

    Returns:
        plt.Figure: Figure Matplotlib générée.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    matrix_to_plot = confusion_matrix.astype("float")
    if normalize:
        row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        matrix_to_plot = matrix_to_plot / row_sums
        fmt = ".2%"
    else:
        fmt = "d"

    sns.heatmap(
        matrix_to_plot if normalize else confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Classe Réelle")
    ax.set_xlabel("Classe Prédite")

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curves(
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[Path | str] = None,
    title: str = "Courbes ROC Comparatives",
) -> plt.Figure:
    """Trace plusieurs courbes ROC sur un même graphique avec leurs AUC respectifs.

    Args:
        roc_curves: Dictionnaire {nom_modèle: (fpr, tpr, roc_auc)}.
        output_path: Chemin où sauvegarder l'image.
        title: Titre du graphique.

    Returns:
        plt.Figure: Figure générée.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (fpr, tpr, auc_val) in roc_curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Aléatoire (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR / Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_models_comparison(
    benchmark_df: Any,
    metric_cols: Sequence[str],
    output_path: Optional[Path | str] = None,
    title: str = "Comparaison des Modèles de Machine Learning",
) -> plt.Figure:
    """Génère un diagramme en barres comparant plusieurs modèles selon différentes métriques.

    Args:
        benchmark_df: DataFrame pandas avec index ou colonne 'model' et colonnes de métriques.
        metric_cols: Liste des colonnes de métriques à afficher.
        output_path: Chemin du fichier image cible.
        title: Titre du graphique.

    Returns:
        plt.Figure: Figure générée.
    """
    set_plot_style()
    import pandas as pd

    df = benchmark_df.copy()
    if "model" not in df.columns:
        df["model"] = df.index

    melted = pd.melt(df, id_vars=["model"], value_vars=list(metric_cols), var_name="Métrique", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="Score", hue="Métrique", ax=ax, palette="mako")
    ax.set_title(title)
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    plt.xticks(rotation=20)

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importances(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 15,
    output_path: Optional[Path | str] = None,
    title: str = "Importance des Variables (Top Features)",
) -> plt.Figure:
    """Trace un barplot horizontal des variables les plus contributrices.

    Args:
        feature_names: Noms des attributs.
        importances: Coefficients d'importance associés.
        top_n: Nombre maximal d'attributs à afficher.
        output_path: Chemin d'enregistrement.
        title: Titre du tracé.

    Returns:
        plt.Figure: Figure générée.
    """
    set_plot_style()
    import pandas as pd

    df = pd.DataFrame({"feature": list(feature_names), "importance": list(importances)})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(data=df, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance Relative")
    ax.set_ylabel("Variable")

    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
