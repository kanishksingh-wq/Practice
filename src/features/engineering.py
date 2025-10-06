import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import timedelta
from category_encoders import TargetEncoder


def add_velocity_features(df: pd.DataFrame, schema: Dict, windows_min: List[int]) -> pd.DataFrame:
    df = df.sort_values([schema['user_id_col'], schema['timestamp_col']]).copy()
    df['timestamp'] = pd.to_datetime(df[schema['timestamp_col']])
    for w in windows_min:
        w_td = pd.to_timedelta(w, unit='m')
        count_col = f'user_tx_count_{w}m'
        sum_col = f'user_amount_sum_{w}m'
        mean_col = f'user_amount_mean_{w}m'
        df[count_col] = np.nan
        df[sum_col] = np.nan
        df[mean_col] = np.nan

        for uid, g in df.groupby(schema['user_id_col'], sort=False):
            idx = g.index
            g_sorted = g.sort_values('timestamp').set_index('timestamp')
            c = g_sorted['transaction_id'].rolling(w_td).count().values
            s = g_sorted['amount'].rolling(w_td).sum().values
            m = g_sorted['amount'].rolling(w_td).mean().values
            df.loc[idx, count_col] = c
            df.loc[idx, sum_col] = s
            df.loc[idx, mean_col] = m
    return df


def add_frequency_features(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    # Frequency encoding for high-cardinality
    for col in [c for c in schema['cat_cols'] if c not in ['country', 'device_type']]:
        freq = df[col].value_counts(dropna=False)
        df[f'{col}_freq'] = df[col].map(freq)
    return df


def add_geo_mismatch(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    if 'device_country' in schema['geo_cols'] and 'country' in schema['geo_cols']:
        df['geo_mismatch'] = (df['device_country'] != df['country']).astype(int)
    else:
        df['geo_mismatch'] = 0
    return df


def add_session_features(df: pd.DataFrame, schema: Dict, session_gap_minutes: int = 30) -> pd.DataFrame:
    df = df.sort_values([schema['user_id_col'], schema['timestamp_col']]).copy()
    gap = pd.to_timedelta(session_gap_minutes, unit='m')
    df['timestamp'] = pd.to_datetime(df[schema['timestamp_col']])
    df['prev_ts'] = df.groupby(schema['user_id_col'])['timestamp'].shift(1)
    df['new_session'] = ((df['timestamp'] - df['prev_ts']) > gap) | df['prev_ts'].isna()
    df['session_id'] = df.groupby(schema['user_id_col'])['new_session'].cumsum()
    sess_grp = df.groupby([schema['user_id_col'], 'session_id'])
    df['session_tx_count'] = sess_grp['transaction_id'].transform('count')
    df['session_amount_sum'] = sess_grp['amount'].transform('sum')
    df['session_amount_mean'] = sess_grp['amount'].transform('mean')
    df.drop(columns=['prev_ts'], inplace=True)
    return df


class TargetEncoderWrapper:
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.enc = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.enc = TargetEncoder(cols=self.cols, smoothing=0.2)
        self.enc.fit(X[self.cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.enc is None:
            raise RuntimeError('TargetEncoderWrapper not fitted')
        te = self.enc.transform(X[self.cols])
        te.columns = [f'{c}_te' for c in te.columns]
        X = pd.concat([X, te], axis=1)
        return X
