import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class Paths:
    data_csv: Optional[str]
    artifacts_dir: str
    logs_dir: str
    mlflow_tracking_uri: str


@dataclass
class Schema:
    label: str
    id_col: str
    user_id_col: str
    timestamp_col: str
    cat_cols: List[str]
    num_cols: List[str]
    geo_cols: List[str]


@dataclass
class Split:
    strategy: str
    train_frac: float
    val_frac: float
    test_frac: float


@dataclass
class Imbalance:
    method: str
    smote_k_neighbors: int


@dataclass
class Features:
    velocity_windows_minutes: List[int]
    session_gap_minutes: int
    target_encoding: bool
    frequency_encoding: bool


@dataclass
class Models:
    families: List[str]
    optuna_trials: int
    n_splits_time_groups: int


@dataclass
class Thresholding:
    optimize_metric: str
    beta: float


@dataclass
class Costs:
    false_positive: float
    false_negative: float


@dataclass
class Shap:
    sample_size: int


@dataclass
class Monitoring:
    psi_bins: int
    drift_reference: str


@dataclass
class Config:
    seed: int
    paths: Paths
    schema: Schema
    split: Split
    imbalance: Imbalance
    features: Features
    models: Models
    thresholding: Thresholding
    costs: Costs
    shap: Shap
    monitoring: Monitoring


def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    paths = Paths(**raw['paths'])
    schema = Schema(**raw['schema'])
    split = Split(**raw['split'])
    imbalance = Imbalance(**raw['imbalance'])
    features = Features(**raw['features'])
    models = Models(**raw['models'])
    thresholding = Thresholding(**raw['thresholding'])
    costs = Costs(**raw['costs'])
    shap_cfg = Shap(**raw['shap'])
    monitoring = Monitoring(**raw['monitoring'])

    # Ensure directories exist at runtime
    os.makedirs(paths.artifacts_dir, exist_ok=True)
    os.makedirs(paths.logs_dir, exist_ok=True)
    os.makedirs(paths.mlflow_tracking_uri, exist_ok=True)

    return Config(
        seed=raw['seed'],
        paths=paths,
        schema=schema,
        split=split,
        imbalance=imbalance,
        features=features,
        models=models,
        thresholding=thresholding,
        costs=costs,
        shap=shap_cfg,
        monitoring=monitoring,
    )
