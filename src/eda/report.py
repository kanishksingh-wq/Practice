import os
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df)) * 100
    out = pd.DataFrame({'column': miss.index, 'missing_count': miss.values, 'missing_pct': pct.values})
    return out.sort_values('missing_pct', ascending=False)


def summarize_duplicates(df: pd.DataFrame, id_col: Optional[str] = None) -> pd.DataFrame:
    if id_col and id_col in df.columns:
        dup = df.duplicated(subset=[id_col]).sum()
    else:
        dup = df.duplicated().sum()
    return pd.DataFrame({'metric': ['duplicate_rows'], 'value': [int(dup)]})


def plot_missingness_bar(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    miss_df = summarize_missingness(df)
    plt.figure(figsize=(10, max(3, len(miss_df) * 0.2)))
    sns.barplot(data=miss_df, x='missing_pct', y='column', color='#1f77b4')
    plt.xlabel('Missing %')
    plt.ylabel('Column')
    plt.title('Missingness by Column')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'missingness_bar.png'), dpi=150)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, num_cols: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for col in num_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=50, kde=False)
        plt.title(f'Distribution: {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'dist_{col}.png'), dpi=120)
        plt.close()


def plot_categorical_topk(df: pd.DataFrame, cat_cols: List[str], out_dir: str, k: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    for col in cat_cols:
        if col not in df.columns:
            continue
        vc = df[col].astype('object').value_counts(dropna=False).head(k).reset_index()
        vc.columns = [col, 'count']
        plt.figure(figsize=(7, 4))
        sns.barplot(data=vc, x='count', y=col, color='#ff7f0e')
        plt.title(f'Top-{k} categories: {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'topk_{col}.png'), dpi=120)
        plt.close()


def plot_time_trends(df: pd.DataFrame, ts_col: str, label_col: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if ts_col not in df.columns:
        return
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    daily = d.set_index(ts_col).resample('D')
    cnt = daily[label_col].count().rename('tx_count')
    fraud_rate = daily[label_col].mean().rename('fraud_rate')
    out = pd.concat([cnt, fraud_rate], axis=1).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(out[ts_col], out['tx_count'], color='#1f77b4', label='tx_count')
    ax1.set_ylabel('Transactions')
    ax2 = ax1.twinx()
    ax2.plot(out[ts_col], out['fraud_rate'], color='#d62728', label='fraud_rate')
    ax2.set_ylabel('Fraud rate')
    fig.suptitle('Daily transactions and fraud rate')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'time_trends.png'), dpi=150)
    plt.close(fig)


def generate_eda(df: pd.DataFrame, id_col: str, ts_col: str, label_col: str, num_cols: List[str], cat_cols: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    miss_df = summarize_missingness(df)
    miss_df.to_csv(os.path.join(out_dir, 'missingness.csv'), index=False)

    dup_df = summarize_duplicates(df, id_col=id_col)
    dup_df.to_csv(os.path.join(out_dir, 'duplicates.csv'), index=False)

    plot_missingness_bar(df, out_dir)
    plot_numeric_distributions(df, num_cols, out_dir)
    plot_categorical_topk(df, cat_cols, out_dir)
    plot_time_trends(df, ts_col, label_col, out_dir)
