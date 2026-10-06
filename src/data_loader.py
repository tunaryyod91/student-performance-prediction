"""Module de chargement, de validation et de partitionnement des données.

Gère le chargement du dataset Student Performance (UCI), le téléchargement automatique
si absent, la création de la variable cible binaire ou multiclasse, et le découpage
stratifié Train/Test garantissant l'absence de fuite d'information.
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UCI_STUDENT_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
RAW_DATA_FILE = DEFAULT_RAW_DIR / "student-mat.csv"


def download_uci_dataset(dest_dir: Path | str = DEFAULT_RAW_DIR) -> Path:
    """Télécharge et extrait l'archive officielle UCI Student Performance.

    Args:
        dest_dir: Dossier de destination pour les fichiers bruts.

    Returns:
        Path: Chemin vers le fichier student-mat.csv extrait.
    """
    destination = ensure_dir(dest_dir)
    target_csv = destination / "student-mat.csv"

    if target_csv.exists() and target_csv.stat().st_size > 0:
        logger.info(f"Le fichier brut existe déjà : {target_csv}")
        return target_csv

    logger.info(f"Téléchargement du dataset depuis {UCI_STUDENT_URL}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(UCI_STUDENT_URL, headers=headers)

    with urllib.request.urlopen(req) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer_zip:
        # L'archive UCI contient un sous-zip nommé student.zip
        if "student.zip" in outer_zip.namelist():
            inner_zip_bytes = outer_zip.read("student.zip")
            with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zip:
                inner_zip.extractall(destination)
        else:
            outer_zip.extractall(destination)

    if not target_csv.exists():
        raise FileNotFoundError(f"Échec de l'extraction de {target_csv}")

    logger.info(f"Dataset extrait avec succès dans {destination}")
    return target_csv


def load_raw_data(
    filepath: Path | str = RAW_DATA_FILE, auto_download: bool = True, sep: str = ";"
) -> pd.DataFrame:
    """Charge le jeu de données brut avec validation d'intégrité.

    Args:
        filepath: Chemin vers student-mat.csv.
        auto_download: Télécharge automatiquement depuis UCI si absent.
        sep: Séparateur de colonnes (point-virgule par défaut pour UCI).

    Returns:
        pd.DataFrame: Données chargées.
    """
    path = Path(filepath)

    if not path.exists():
        if auto_download:
            logger.warning(f"Fichier introuvable à {path}. Tentative de téléchargement automatique...")
            path = download_uci_dataset(path.parent)
        else:
            raise FileNotFoundError(
                f"Fichier {path} introuvable. Placez 'student-mat.csv' dans 'data/raw/' "
                f"ou activez auto_download=True."
            )

    df = pd.read_csv(path, sep=sep)
    logger.info(f"Données brutes chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Valide les colonnes minimales requises dans le dataset étudiant."""
    required_cols = {"school", "sex", "age", "studytime", "failures", "absences", "G3"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes dans le dataset : {missing}")


def prepare_features_and_target(
    df: pd.DataFrame,
    target_type: str = "binary",
    include_intermediate_grades: bool = False,
    pass_threshold: int = 10,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables prédictives (X) de la variable cible (y).

    La variable G3 (note finale sur 20) est convertie selon l'objectif :
      - 'binary' : 1 si G3 >= pass_threshold (succès académique), 0 sinon.
      - 'multiclass' : 3 niveaux : Fail (<10), Pass (10-13), High (>=14).
      - 'regression' : Valeur brute de G3 (note continue).

    Args:
        df: DataFrame original.
        target_type: 'binary', 'multiclass', ou 'regression'.
        include_intermediate_grades: Si False (recommandé pour étudier l'impact socio-éducatif pur),
                                     G1 et G2 sont retirées pour éviter une prédiction triviale.
        pass_threshold: Seuil de validation en notation sur 20 (par défaut 10).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X et variable cible y.
    """
    validate_raw_schema(df)
    data = df.copy()

    # Création de la cible
    if target_type == "binary":
        y = (data["G3"] >= pass_threshold).astype(int)
        y.name = "passed"
    elif target_type == "multiclass":

        def categorize_grade(score: int) -> str:
            if score < pass_threshold:
                return "Fail"
            elif score <= 13:
                return "Pass"
            else:
                return "High"

        y = data["G3"].apply(categorize_grade)
        y.name = "grade_group"
    elif target_type == "regression":
        y = data["G3"]
        y.name = "G3"
    else:
        raise ValueError(f"Type de cible non reconnu : {target_type}")

    # Gestion des variables explicatives
    cols_to_drop = ["G3"]
    if not include_intermediate_grades:
        # G1 et G2 sont fortement corrélées à G3 et peuvent masquer l'influence socio-démographique
        for col in ["G1", "G2"]:
            if col in data.columns:
                cols_to_drop.append(col)

    X = data.drop(columns=cols_to_drop)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Effectue un découpage Train/Test étanche et stratifié.

    Args:
        X: Variables indépendantes.
        y: Variable dépendante.
        test_size: Proportion de l'échantillon test (défaut 0.2).
        random_state: Graine aléatoire de reproductibilité.
        stratify: Si True, préserve la distribution des classes.

    Returns:
        Tuple[X_train, X_test, y_train, y_test]
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    logger.info(
        f"Partitionnement Train/Test terminé (test_size={test_size}) :\n"
        f"  - X_train : {X_train.shape}, y_train : {y_train.shape}\n"
        f"  - X_test  : {X_test.shape}, y_test  : {y_test.shape}"
    )
    return X_train, X_test, y_train, y_test


def save_processed_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> None:
    """Sauvegarde les découpages dans le dossier data/processed/.

    Args:
        X_train, X_test: Données explicatives.
        y_train, y_test: Cibles associées.
        output_dir: Dossier de sauvegarde.
    """
    out_path = ensure_dir(output_dir)
    X_train.to_csv(out_path / "X_train.csv", index=False)
    X_test.to_csv(out_path / "X_test.csv", index=False)
    y_train.to_frame().to_csv(out_path / "y_train.csv", index=False)
    y_test.to_frame().to_csv(out_path / "y_test.csv", index=False)
    logger.info(f"Fichiers exportés dans {out_path}")


def main() -> None:
    """Point d'entrée CLI pour la gestion des données."""
    parser = argparse.ArgumentParser(description="Gestion et chargement du dataset Student Performance.")
    parser.add_argument(
        "--download", action="store_true", help="Télécharge le dataset officiel UCI dans data/raw/."
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "regression"],
        help="Type de cible à créer (binary par défaut).",
    )
    parser.add_argument(
        "--include-grades",
        action="store_true",
        help="Inclure G1 et G2 dans les variables d'entrée.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Génère et sauvegarde les découpages X_train, X_test, y_train, y_test.",
    )
    args = parser.parse_args()

    if args.download or not RAW_DATA_FILE.exists():
        download_uci_dataset()

    df = load_raw_data(RAW_DATA_FILE)
    print("\n=== APERÇU DU DATASET BRUT ===")
    print(df.info())
    print("\n5 premières lignes :")
    print(df.head())

    X, y = prepare_features_and_target(
        df, target_type=args.target_type, include_intermediate_grades=args.include_grades
    )
    print(f"\nDistribution de la variable cible ({y.name}) :")
    print(y.value_counts(normalize=True).apply(lambda v: f"{v*100:.2f}%"))

    if args.split:
        X_train, X_test, y_train, y_test = split_data(X, y)
        save_processed_splits(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
