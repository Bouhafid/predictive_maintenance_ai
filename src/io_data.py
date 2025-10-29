from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import pandas as pd

# Colonnes C-MAPSS : id, cycle, 3 op settings, 21 capteurs
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_set_{i}" for i in range(1, 4)]
    + [f"s_{i:02d}" for i in range(1, 22)]
)

def _read_fd_txt(path: str) -> pd.DataFrame:
    """Lit un fichier C-MAPSS (train/test) à colonnes séparées par espaces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLUMNS
    return df

def _read_rul_txt(path: str) -> pd.DataFrame:
    """Lit RUL_FDxxx.txt et crée un index engine_id 1..N aligné au test."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    rul = pd.read_csv(path, sep=r"\s+", header=None)
    rul.columns = ["RUL_truth"]
    rul.index = np.arange(1, len(rul) + 1)
    return rul

def _add_rul_labels(
    train_df: pd.DataFrame, test_df: pd.DataFrame, rul_truth: pd.DataFrame, clip: int = 125
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcule la RUL pour train et test et ajoute la colonne RUL (clippée)."""
    train_df = train_df.copy()
    test_df  = test_df.copy()

    # Train: RUL = max(cycle)_engine - cycle
    rul_train = train_df.groupby("engine_id")["cycle"].transform("max") - train_df["cycle"]
    train_df["RUL"] = rul_train.clip(upper=clip)

    # Test: max_total = cycle_last(test) + RUL_truth ; RUL = max_total - cycle
    last_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    last_cycles = last_cycles.merge(rul_truth, left_on="engine_id", right_index=True, how="left")
    last_cycles["max_total"] = last_cycles["cycle"] + last_cycles["RUL_truth"]

    test_df = test_df.merge(last_cycles[["engine_id","max_total"]], on="engine_id", how="left")
    test_df["RUL"] = (test_df["max_total"] - test_df["cycle"]).clip(upper=clip)
    test_df.drop(columns=["max_total"], inplace=True)

    return train_df, test_df

def load_fd_dataset(root: str, dataset: str, rul_clip: int = 125) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge un sous-ensemble C-MAPSS (FD001..FD004) depuis root (ex: data/raw),
    calcule la RUL et retourne (train_df, test_df).
    """
    base = os.path.join(root, dataset)
    train_path = os.path.join(base, f"train_{dataset}.txt")
    test_path  = os.path.join(base, f"test_{dataset}.txt")
    rul_path   = os.path.join(base, f"RUL_{dataset}.txt")

    train = _read_fd_txt(train_path)
    test  = _read_fd_txt(test_path)
    rul   = _read_rul_txt(rul_path)

    train, test = _add_rul_labels(train, test, rul, clip=rul_clip)
    return train, test

def save_processed(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde un DataFrame pré-processé en Parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
