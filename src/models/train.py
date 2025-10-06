import os
import json
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


ALGO_FNS = {}


def _scale_pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return max((neg / max(pos, 1)) - 1, 1)


def get_sampler(method: str, k_neighbors: int = 5):
    if method == 'smote':
        return SMOTE(k_neighbors=k_neighbors, random_state=42)
    if method == 'adasyn':
        return ADASYN(n_neighbors=k_neighbors, random_state=42)
    return None


def time_groups_from_timestamp(ts: pd.Series, n_groups: int = 3) -> np.ndarray:
    # Map timestamps to quantile-based groups for time-aware CV
    t = pd.to_datetime(ts)
    try:
        t_int = t.view('int64')
    except Exception:
        t_int = t.astype('int64')
    q = pd.qcut(t_int, q=min(n_groups, max(2, t.nunique())), labels=False, duplicates='drop')
    return q.values


def optuna_objective(factory_fn, X, y, groups, imbalance_method: str, smote_k: int):
    def objective(trial: optuna.Trial):
        model, params, supports_weight = factory_fn(trial)
        sampler = get_sampler(imbalance_method, smote_k)
        n_unique = len(np.unique(groups))
        n_splits = int(max(2, min(5, n_unique)))
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(y))
        for train_idx, valid_idx in gkf.split(X, y, groups):
            X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

            if sampler is not None:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

            if supports_weight:
                spw = _scale_pos_weight(y_tr)
                if 'xgb' in model.__class__.__name__.lower():
                    model.set_params(scale_pos_weight=spw)
                elif 'lgbm' in model.__class__.__name__.lower():
                    model.set_params(scale_pos_weight=spw)
                elif 'catboost' in model.__class__.__name__.lower():
                    model.set_params(scale_pos_weight=spw)

            model.fit(X_tr, y_tr)
            oof[valid_idx] = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y, oof)
    return objective


def model_factory_lightgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
    }
    model = LGBMClassifier(**params)
    return model, params, True


def model_factory_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'use_label_encoder': False,
    }
    model = XGBClassifier(**params)
    return model, params, True


def model_factory_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_state': 42,
        'eval_metric': 'AUC',
        'verbose': False,
    }
    model = CatBoostClassifier(**params)
    return model, params, True


def model_factory_mlp(trial):
    params = {
        'hidden_layer_sizes': (trial.suggest_int('h1', 32, 128), trial.suggest_int('h2', 16, 64)),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'max_iter': 50,
        'random_state': 42,
    }
    model = MLPClassifier(**params)
    return model, params, False


FACTORIES = {
    'lightgbm': model_factory_lightgbm,
    'xgboost': model_factory_xgboost,
    'catboost': model_factory_catboost,
    'mlp': model_factory_mlp,
}


def train_and_select(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                     families: List[str], optuna_trials: int, time_groups: np.ndarray,
                     imbalance_method: str, smote_k: int, artifacts_dir: str,
                     mlflow_tracking_uri: Optional[str] = None,
                     experiment_name: str = 'fraud_detection') -> Dict:
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    results = {}
    for fam in families:
        factory = FACTORIES[fam]
        study = optuna.create_study(direction='maximize')
        objective = optuna_objective(factory, X_train, y_train, time_groups, imbalance_method, smote_k)
        study.optimize(objective, n_trials=optuna_trials)

        best_model, best_params, supports_weight = factory(optuna.trial.FixedTrial(study.best_params))
        sampler = get_sampler(imbalance_method, smote_k)
        X_fit, y_fit = X_train, y_train
        if sampler is not None:
            X_fit, y_fit = sampler.fit_resample(X_fit, y_fit)
        if supports_weight:
            spw = _scale_pos_weight(y_fit)
            if 'xgb' in best_model.__class__.__name__.lower():
                best_model.set_params(scale_pos_weight=spw)
            elif 'lgbm' in best_model.__class__.__name__.lower():
                best_model.set_params(scale_pos_weight=spw)
            elif 'catboost' in best_model.__class__.__name__.lower():
                best_model.set_params(scale_pos_weight=spw)

        best_model.fit(X_fit, y_fit)
        val_prob = best_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        results[fam] = {
            'model': best_model,
            'params': best_params,
            'val_auc': float(val_auc),
            'val_prob': val_prob.tolist(),
        }

        # Save params
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, f'{fam}_best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)

        # Log to MLflow
        with mlflow.start_run(run_name=f'{fam}_best'):
            mlflow.log_params({f'{fam}_{k}': v for k, v in best_params.items()})
            mlflow.log_metric('val_auc', float(val_auc))
            try:
                mlflow.sklearn.log_model(best_model, artifact_path=f'{fam}_model')
            except Exception:
                pass

    return results
