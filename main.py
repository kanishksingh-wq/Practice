import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.config import load_config
from src.utils.logging_utils import setup_logging
from src.data.synthetic import generate_synthetic
from src.data.preprocess import Preprocessor, temporal_split
from src.features.engineering import add_velocity_features, add_frequency_features, add_geo_mismatch, add_session_features, TargetEncoderWrapper
from src.features.selection import select_by_mutual_info
from src.models.train import train_and_select, time_groups_from_timestamp
from src.evaluation.metrics import compute_metrics, find_best_threshold, expected_cost
from src.explainability.shap_utils import shap_global_plots
from src.monitoring.drift import compute_psi_frame
from src.eda.report import generate_eda


def main(config_path: str = 'configs/config.yaml'):
    cfg = load_config(config_path)
    logger = setup_logging(cfg.paths.logs_dir)
    logger.info('Loaded configuration')

    # Load or generate data
    if cfg.paths.data_csv and os.path.exists(cfg.paths.data_csv):
        df = pd.read_csv(cfg.paths.data_csv, parse_dates=[cfg.schema.timestamp_col])
        logger.info(f'Loaded dataset from {cfg.paths.data_csv} shape={df.shape}')
    else:
        df = generate_synthetic(seed=cfg.seed)
        logger.info(f'Generated synthetic dataset shape={df.shape}')

    # EDA artifacts
    eda_dir = os.path.join(cfg.paths.artifacts_dir, 'eda')
    try:
        generate_eda(df, id_col=cfg.schema.id_col, ts_col=cfg.schema.timestamp_col, label_col=cfg.schema.label,
                     num_cols=cfg.schema.num_cols, cat_cols=cfg.schema.cat_cols, out_dir=eda_dir)
        logger.info('Saved EDA artifacts to %s', eda_dir)
    except Exception as e:
        logger.warning('EDA generation failed: %s', e)

    # Basic cleaning
    pre = Preprocessor(schema={
        'id_col': cfg.schema.id_col,
        'num_cols': cfg.schema.num_cols,
        'cat_cols': cfg.schema.cat_cols,
    }, seed=cfg.seed)
    df = pre.drop_duplicates(df)
    df = df.sort_values(cfg.schema.timestamp_col)

    # Feature engineering (pre-imputation for consistent rolling ops)
    df = df.copy()
    df = add_geo_mismatch(df, schema={'geo_cols': cfg.schema.geo_cols})
    df = add_frequency_features(df, schema={'cat_cols': cfg.schema.cat_cols})
    df = add_session_features(df, schema={
        'user_id_col': cfg.schema.user_id_col,
        'timestamp_col': cfg.schema.timestamp_col
    }, session_gap_minutes=cfg.features.session_gap_minutes)
    df = add_velocity_features(df, schema={
        'user_id_col': cfg.schema.user_id_col,
        'timestamp_col': cfg.schema.timestamp_col
    }, windows_min=cfg.features.velocity_windows_minutes)

    # Select columns for modeling
    label = cfg.schema.label
    feat_cols = (
        cfg.schema.num_cols +
        cfg.schema.cat_cols +
        [
            'geo_mismatch', 'session_tx_count', 'session_amount_sum', 'session_amount_mean'
        ] +
        [c for c in df.columns if c.endswith('_freq') or c.startswith('user_tx_count_') or c.startswith('user_amount_')]
    )
    feat_cols = [c for c in feat_cols if c in df.columns]

    # Split
    train_df, val_df, test_df = temporal_split(df, schema={'timestamp_col': cfg.schema.timestamp_col},
                                               train_frac=cfg.split.train_frac, val_frac=cfg.split.val_frac, test_frac=cfg.split.test_frac)

    # Fit preprocessing on train
    pre = Preprocessor(schema={
        'id_col': cfg.schema.id_col,
        'num_cols': [c for c in feat_cols if c in cfg.schema.num_cols or c.startswith('session_') or c.startswith('user_') or c.endswith('_freq') or c == 'geo_mismatch'],
        'cat_cols': [c for c in feat_cols if c in cfg.schema.cat_cols],
    }, seed=cfg.seed)

    X_tr = pre.fit_transform(train_df[feat_cols])
    X_va = pre.transform(val_df[feat_cols])
    X_te = pre.transform(test_df[feat_cols])

    y_tr = train_df[label].astype(int)
    y_va = val_df[label].astype(int)
    y_te = test_df[label].astype(int)

    # Target encoding on categorical
    if cfg.features.target_encoding and len(pre.schema['cat_cols']) > 0:
        tew = TargetEncoderWrapper(cols=pre.schema['cat_cols'])
        tew.fit(X_tr, y_tr)
        X_tr = tew.transform(X_tr)
        X_va = tew.transform(X_va)
        X_te = tew.transform(X_te)
        # Drop raw categorical columns after target encoding to ensure numeric-only features
        X_tr = X_tr.drop(columns=pre.schema['cat_cols'], errors='ignore')
        X_va = X_va.drop(columns=pre.schema['cat_cols'], errors='ignore')
        X_te = X_te.drop(columns=pre.schema['cat_cols'], errors='ignore')

    # Feature selection (optional, MI)
    selected = select_by_mutual_info(X_tr, y_tr, k=min(100, X_tr.shape[1]))
    X_tr = X_tr[selected]
    X_va = X_va[selected]
    X_te = X_te[selected]

    # Train models with Optuna
    groups = time_groups_from_timestamp(train_df[cfg.schema.timestamp_col], n_groups=cfg.models.n_splits_time_groups)
    results = train_and_select(X_tr, y_tr, X_va, y_va,
                               families=cfg.models.families,
                               optuna_trials=cfg.models.optuna_trials,
                               time_groups=groups,
                               imbalance_method=cfg.imbalance.method,
                               smote_k=cfg.imbalance.smote_k_neighbors,
                               artifacts_dir=cfg.paths.artifacts_dir,
                               mlflow_tracking_uri=cfg.paths.mlflow_tracking_uri,
                               experiment_name='fraud_detection')

    # Evaluate and thresholding
    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    metrics_out = {}
    best_family, best_auc = None, -np.inf
    for fam, info in results.items():
        model = info['model']
        va_prob = np.array(info['val_prob'])
        thr, _ = find_best_threshold(y_va, va_prob, metric=cfg.thresholding.optimize_metric,
                                     beta=cfg.thresholding.beta,
                                     cost_fp=cfg.costs.false_positive,
                                     cost_fn=cfg.costs.false_negative)
        te_prob = model.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, te_prob, threshold=thr)
        y_pred = (te_prob >= thr).astype(int)
        m['threshold'] = float(thr)
        m['expected_cost'] = float(expected_cost(y_te, y_pred, cfg.costs.false_positive, cfg.costs.false_negative))
        m['val_auc'] = float(info['val_auc'])
        m['test_auc'] = float(roc_auc_score(y_te, te_prob))
        metrics_out[fam] = m
        if info['val_auc'] > best_auc:
            best_family, best_auc = fam, info['val_auc']

    with open(os.path.join(cfg.paths.artifacts_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Explainability using SHAP for best model
    best_model = results[best_family]['model']
    sample_idx = np.random.choice(len(X_tr), size=min(cfg.shap.sample_size, len(X_tr)), replace=False)
    shap_global_plots(best_model, X_tr, X_tr.iloc[sample_idx], cfg.paths.artifacts_dir, prefix=best_family)

    # Drift monitoring (PSI) between train and test
    psi_scores = compute_psi_frame(pd.DataFrame(X_tr, columns=X_tr.columns), pd.DataFrame(X_te, columns=X_te.columns), features=X_tr.columns, bins=10)
    with open(os.path.join(cfg.paths.artifacts_dir, 'psi.json'), 'w') as f:
        json.dump(psi_scores, f, indent=2)

    logger.info('Pipeline complete. Artifacts saved to %s', cfg.paths.artifacts_dir)


if __name__ == '__main__':
    main()
