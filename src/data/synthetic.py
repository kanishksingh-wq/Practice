import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Tuple


def generate_synthetic(n_users: int = 10000, n_tx: int = 300000, start_date: str = '2023-01-01', seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1)
    user_account_age = rng.integers(1, 3650, size=n_users)
    user_device_trust = rng.uniform(0.0, 1.0, size=n_users)
    user_country = rng.choice(['US','GB','IN','DE','FR','CA','AU'], size=n_users, p=[0.3,0.1,0.2,0.1,0.1,0.1,0.1])

    users = pd.DataFrame({
        'user_id': user_ids,
        'account_age_days': user_account_age,
        'device_trust_score': user_device_trust,
        'device_country': user_country,
    })

    timestamps = pd.date_range(start_date, periods=365, freq='D')
    merchant_ids = [f'm_{i}' for i in range(5000)]
    merchant_cats = [f'c_{i}' for i in range(200)]
    device_ids = [f'd_{i}' for i in range(20000)]

    df = pd.DataFrame({
        'transaction_id': np.arange(1, n_tx + 1),
        'user_id': rng.choice(user_ids, size=n_tx),
        'timestamp': rng.choice(timestamps, size=n_tx),
        'amount': np.round(rng.gamma(2.0, 50.0, size=n_tx), 2),
        'merchant_id': rng.choice(merchant_ids, size=n_tx),
        'merchant_category': rng.choice(merchant_cats, size=n_tx),
        'country': rng.choice(['US','GB','IN','DE','FR','CA','AU'], size=n_tx, p=[0.3,0.1,0.2,0.1,0.1,0.1,0.1]),
        'device_id': rng.choice(device_ids, size=n_tx),
        'device_type': rng.choice(['mobile','web','tablet'], size=n_tx, p=[0.6,0.35,0.05]),
    })

    df = df.merge(users, on='user_id', how='left')

    # Fraud generation: base rate with risk factors
    base_logit = -4.0
    risk = (
        0.0008 * df['amount'] +
        0.5 * (df['country'] != df['device_country']).astype(int) +
        0.4 * (df['device_trust_score'] < 0.2).astype(int) +
        0.3 * (df['account_age_days'] < 60).astype(int)
    )
    fraud_prob = 1 / (1 + np.exp(-(base_logit + risk)))
    df['is_fraud'] = (rng.random(n_tx) < fraud_prob).astype(int)

    # Missingness and duplicates to mimic real data
    mask_missing = rng.random(n_tx) < 0.01
    df.loc[mask_missing, 'merchant_category'] = np.nan
    dup_idx = rng.choice(df.index, size=int(0.002 * n_tx), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Shuffle timestamps slightly within the day
    df['timestamp'] = pd.to_datetime(df['timestamp']) + pd.to_timedelta(rng.integers(0, 24*60, size=len(df)), unit='m')

    return df
