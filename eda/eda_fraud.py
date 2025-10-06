import os
import glob
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_categorical_dtype,
)
from IPython.display import display
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------
# Utility helpers
# ----------------------------
RANDOM_STATE = 42


def _rng():
    return np.random.default_rng(RANDOM_STATE)


def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=RANDOM_STATE)


def lower_map(cols: List[str]) -> Dict[str, str]:
    return {c: c.lower() for c in cols}


def unique_ratio(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    try:
        u = s.nunique(dropna=True)
    except Exception:
        u = len(pd.unique(s.dropna()))
    return u / max(1, n)


def is_binary_series(s: pd.Series) -> bool:
    try:
        vals = pd.unique(s.dropna())
        return len(vals) == 2
    except Exception:
        return False


def find_by_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    prefer_numeric: bool = False,
    binary: bool = False,
    prefer_object: bool = False,
) -> List[str]:
    cols = list(df.columns)
    low = lower_map(cols)
    matches = []
    for c in cols:
        lc = low[c]
        if any(k in lc for k in keywords):
            matches.append(c)
    # filter by dtype preference
    if prefer_numeric:
        matches = [c for c in matches if is_numeric_dtype(df[c])] or matches
    if prefer_object:
        matches = [c for c in matches if (df[c].dtype == "object" or is_categorical_dtype(df[c]))] or matches
    if binary:
        matches = [c for c in matches if is_binary_series(df[c])] or matches
    return matches


# ----------------------------
# Column inference
# ----------------------------

def detect_target(df: pd.DataFrame, hints: Dict[str, Optional[str]]) -> Optional[str]:
    if hints and hints.get("target") in df.columns:
        return hints["target"]
    cand = find_by_keywords(df, ["fraud", "is_fraud", "label", "target", "y", "scam", "isfraud"], binary=True)
    if cand:
        return cand[0]
    for c in df.columns:
        if is_binary_series(df[c]):
            return c
    return None


def detect_amount(df: pd.DataFrame, hints: Dict[str, Optional[str]]) -> Optional[str]:
    if hints and hints.get("amount") in df.columns:
        return hints["amount"]
    cand = find_by_keywords(df, ["amount", "amt", "price", "value", "transaction_amount"], prefer_numeric=True)
    if cand:
        return cand[0]
    num = [c for c in df.columns if is_numeric_dtype(df[c])]
    if not num:
        return None
    scored: List[Tuple[float, str]] = []
    for c in num:
        s = df[c]
        med = s.median(skipna=True)
        var = s.var(skipna=True)
        if pd.isna(var):
            continue
        score = (0 if pd.isna(med) else (1 if med > 0 else 0)) + (var if np.isfinite(var) else 0)
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def detect_timestamp(df: pd.DataFrame, hints: Dict[str, Optional[str]]) -> Optional[str]:
    if hints and hints.get("timestamp") in df.columns:
        return hints["timestamp"]
    cand = find_by_keywords(df, ["time", "date", "timestamp", "datetime"])
    if cand:
        return cand[0]
    return None


def detect_cols_by_keywords(df: pd.DataFrame, hints: Dict[str, Optional[str]], key_map: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, kws in key_map.items():
        hint = hints.get(name) if hints else None
        if hint in df.columns:
            out[name] = hint
            continue
        found = find_by_keywords(df, kws)
        out[name] = found[0] if found else None
    return out


def is_id_like(df: pd.DataFrame, col: Optional[str]) -> bool:
    if col is None or col not in df.columns:
        return False
    ur = unique_ratio(df[col])
    return ur > 0.98


def encode_binary_target(s: pd.Series) -> Tuple[pd.Series, object]:
    x = s.dropna().unique()
    if len(x) != 2:
        raise ValueError("Target is not binary.")
    vc = s.value_counts(dropna=True)
    labels_sorted = list(vc.sort_values().index)  # rare first
    positive = labels_sorted[0]
    mapping = {labels_sorted[1]: 0, labels_sorted[0]: 1}
    try:
        return s.map(mapping).astype("int8"), positive
    except Exception:
        return pd.Series([mapping.get(v, np.nan) for v in s], index=s.index).astype("float"), positive


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def build_feature_groups(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, List[str]]:
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if (df[c].dtype == "object" or is_categorical_dtype(df[c]))]
    datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    id_like = [c for c in df.columns if is_id_like(df, c)]
    numeric_cols = [c for c in numeric_cols if c not in id_like]
    categorical_cols = [c for c in categorical_cols if c not in id_like]
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "id_like": id_like,
    }


def infer_columns(df: pd.DataFrame, column_hints: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Optional[str]]:
    hints = column_hints or {}
    cols: Dict[str, Optional[str]] = {}
    cols["target"] = detect_target(df, hints)
    cols["amount"] = detect_amount(df, hints)
    cols["timestamp"] = detect_timestamp(df, hints)
    other = detect_cols_by_keywords(
        df,
        hints,
        key_map={
            "merchant_category": ["merchant_category", "mcc", "category", "merchant_cat"],
            "merchant_id": ["merchant_id", "merchant", "store_id"],
            "device_type": ["device_type", "device"],
            "device_os": ["device_os", "os", "platform"],
            "ip_address": ["ip", "ip_address"],
            "user_id": ["user_id", "uid", "customer_id", "account_id"],
            "user_age": ["age"],
            "user_gender": ["gender", "sex"],
            "user_income": ["income"],
            "latitude": ["lat", "latitude"],
            "longitude": ["lon", "lng", "longitude"],
            "city": ["city"],
            "state": ["state", "region", "province"],
            "country": ["country", "nation"],
            "transaction_id": ["transaction_id", "tx_id", "tid", "event_id"],
        },
    )
    cols.update(other)
    # Apply explicit hints to override inference
    for k, v in (hints or {}).items():
        if v in df.columns:
            cols[k] = v
    return cols


# ----------------------------
# Data loading
# ----------------------------

def auto_find_dataset() -> Optional[str]:
    patterns = ["*.csv", "*transactions*.csv", "*fraud*.csv", "*.parquet"]
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
        candidates.extend(glob.glob(os.path.join("data", pat)))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


def load_dataset(dataset_path: Optional[str] = None, auto_find_data: bool = True) -> Tuple[pd.DataFrame, Optional[str]]:
    path = dataset_path
    if not path and auto_find_data:
        path = auto_find_dataset()
    if not path:
        raise FileNotFoundError("No dataset path provided and auto-discovery found nothing.")
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext in [".csv", ""]:
        df = pd.read_csv(path, low_memory=False)
    elif ext in [".parquet"]:
        df = pd.read_parquet(path)
    elif ext in [".feather"]:
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df, path


# ----------------------------
# EDA sections
# ----------------------------

def overview(df: pd.DataFrame):
    print("Shape:", df.shape)
    display(df.sample(n=min(5, len(df)), random_state=RANDOM_STATE))
    print("\nDtypes:")
    display(df.dtypes.to_frame("dtype"))
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Approx. memory usage: {mem_mb:.2f} MB")
    nunique = df.nunique(dropna=True)
    display(nunique.sort_values(ascending=False).to_frame("nunique").head(20))


def missingness(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    display(miss.to_frame("missing_rate").style.format({"missing_rate": "{:.2%}"}))
    top_missing = miss[miss > 0].head(30)
    if not top_missing.empty:
        plt.figure(figsize=(10, max(3, 0.3 * len(top_missing))))
        sns.barplot(x=top_missing.values, y=top_missing.index, color="#4C78A8")
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
        plt.title("Top missingness per column")
        plt.xlabel("Missing rate")
        plt.ylabel("Column")
        plt.tight_layout()
        plt.show()

    try:
        import missingno as msno  # type: ignore

        dfm = sample_df(df, 3000)
        msno.matrix(dfm)
        plt.show()
    except Exception:
        dfm = sample_df(df, 3000)
        plt.figure(figsize=(12, 6))
        sns.heatmap(dfm.isna(), cbar=False)
        plt.title("Missingness heatmap (sample)")
        plt.tight_layout()
        plt.show()


def class_imbalance(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[Optional[pd.Series], Optional[object]]:
    if not target_col or target_col not in df.columns or not is_binary_series(df[target_col]):
        print("Skipping class imbalance: no binary target detected.")
        return None, None
    y_raw = df[target_col]
    y, positive_label = encode_binary_target(y_raw)
    df["__y__"] = y
    vc = y.value_counts()
    rate = float(y.mean())
    print(f"Positive (\"{positive_label}\") rate: {rate:.4f} ({rate*100:.2f}%)")
    display(vc.to_frame("count"))
    plt.figure(figsize=(5, 3))
    sns.barplot(x=vc.index.astype(str), y=vc.values, palette=["#4C78A8", "#F58518"])
    plt.title("Class counts")
    plt.xlabel("Class (0/1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    maj_acc = max(rate, 1 - rate)
    print(f"Majority-class baseline accuracy: {maj_acc:.4f}")
    return y, positive_label


def numeric_distributions_and_outliers(df: pd.DataFrame, inferred: Dict[str, Optional[str]], features: Dict[str, List[str]]):
    amount_col = inferred.get("amount")
    target_available = "__y__" in df.columns
    numeric_cols = features.get("numeric", [])

    if amount_col in df.columns and is_numeric_dtype(df[amount_col]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[amount_col].dropna(), kde=True, ax=axes[0], color="#4C78A8")
        axes[0].set_title(f"{amount_col} distribution")
        sns.histplot(np.log1p(df[amount_col].clip(lower=0)).dropna(), kde=True, ax=axes[1], color="#F58518")
        axes[1].set_title(f"log1p({amount_col}) distribution")
        plt.tight_layout(); plt.show()
        if target_available:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df[[amount_col, "__y__"]].dropna(), x="__y__", y=amount_col)
            plt.title(f"{amount_col} by class")
            plt.xlabel("Class (0/1)")
            plt.tight_layout(); plt.show()

    def outlier_iqr(s: pd.Series) -> float:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return 0.0
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        m = (s < lo) | (s > hi)
        return float(m.mean())

    outlier_rates: List[Tuple[float, str]] = []
    for c in numeric_cols:
        try:
            rate = outlier_iqr(df[c].dropna())
            outlier_rates.append((rate, c))
        except Exception:
            pass
    outlier_rates.sort(reverse=True)
    if outlier_rates:
        disp = pd.DataFrame(outlier_rates, columns=["outlier_rate", "column"])
        disp["outlier_rate"] = disp["outlier_rate"].map(lambda x: f"{x:.2%}")
        display(disp.head(20))


def categorical_distributions(df: pd.DataFrame, features: Dict[str, List[str]]):
    cat_cols = features.get("categorical", [])
    candidates: List[Tuple[int, str]] = []
    for c in cat_cols:
        try:
            k = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        if 2 <= k <= 50:
            candidates.append((k, c))
    candidates.sort(reverse=True)
    sel = [c for _, c in candidates[:8]]

    for col in sel:
        tmp = df[[col]].copy()
        if "__y__" in df.columns:
            tmp["__y__"] = df["__y__"]
        vc = tmp[col].value_counts(dropna=False).head(20)
        plt.figure(figsize=(10, max(3, 0.35 * len(vc))))
        sns.barplot(x=vc.values, y=vc.index.astype(str), color="#4C78A8")
        plt.title(f"{col} top levels")
        plt.xlabel("Count"); plt.ylabel(col)
        plt.tight_layout(); plt.show()

        if "__y__" in tmp.columns:
            gr = tmp.groupby(col)["__y__"].agg(["count", "mean"]).sort_values("mean", ascending=False).head(20)
            fig, ax1 = plt.subplots(figsize=(10, max(3, 0.35 * len(gr))))
            sns.barplot(x=gr["mean"], y=gr.index.astype(str), ax=ax1, color="#F58518")
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1%}"))
            ax1.set_title(f"Fraud rate by {col} (top 20 by rate)")
            ax1.set_xlabel("Fraud rate"); ax1.set_ylabel(col)
            plt.tight_layout(); plt.show()


def correlation_analysis(df: pd.DataFrame, features: Dict[str, List[str]]):
    num_cols = features.get("numeric", [])
    if len(num_cols) >= 2:
        dfc = sample_df(df[num_cols], 200000)
        corr = dfc.corr(method="pearson")
        plt.figure(figsize=(min(1 + 0.4 * len(corr), 14), min(1 + 0.4 * len(corr), 14)))
        sns.heatmap(corr, cmap="vlag", center=0, square=True)
        plt.title("Numeric correlation heatmap (Pearson)")
        plt.tight_layout(); plt.show()

    if "__y__" in df.columns and num_cols:
        rows = []
        for c in num_cols:
            s = df[c]
            y = df["__y__"]
            valid = s.notna() & y.notna()
            if valid.sum() < 100:
                r = np.nan
                p = np.nan
            else:
                try:
                    r, p = stats.pointbiserialr(y[valid], s[valid])
                except Exception:
                    r, p = np.nan, np.nan
            rows.append((c, r, p))
        tb = pd.DataFrame(rows, columns=["feature", "pointbiserial_r", "p_value"]).sort_values("pointbiserial_r", ascending=False)
        display(tb.head(20))


def temporal_analysis(df: pd.DataFrame, inferred: Dict[str, Optional[str]]):
    ts_col = inferred.get("timestamp")
    if not ts_col or ts_col not in df.columns:
        print("Skipping temporal analysis: no timestamp column.")
        return
    if not is_datetime64_any_dtype(df[ts_col]):
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        except Exception:
            print("Timestamp parsing failed; skipping temporal analysis.")
            return
    dft = df[[ts_col]].copy()
    if "__y__" in df.columns:
        dft["__y__"] = df["__y__"]
    dft["hour"] = dft[ts_col].dt.hour
    dft["dow"] = dft[ts_col].dt.dayofweek
    dft["date"] = dft[ts_col].dt.date

    # Hourly counts and fraud rate
    gr = dft.groupby("hour").agg(count=(ts_col, "count"), fraud_rate=("__y__", "mean"))
    gr = gr.fillna({"fraud_rate": 0})
    fig, ax1 = plt.subplots(figsize=(9, 4))
    sns.barplot(x=gr.index, y=gr["count"], color="#4C78A8", ax=ax1)
    ax1.set_title("Transactions by hour")
    ax1.set_xlabel("Hour"); ax1.set_ylabel("Count")
    plt.tight_layout(); plt.show()

    if "__y__" in dft.columns:
        plt.figure(figsize=(9, 3.5))
        sns.lineplot(x=gr.index, y=gr["fraud_rate"], marker="o", color="#F58518")
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1%}"))
        plt.title("Fraud rate by hour")
        plt.xlabel("Hour"); plt.ylabel("Fraud rate")
        plt.tight_layout(); plt.show()

    # Daily trend
    grd = dft.groupby("date").agg(count=(ts_col, "count"), fraud_rate=("__y__", "mean")).fillna({"fraud_rate": 0})
    if len(grd) > 1:
        fig, ax1 = plt.subplots(figsize=(10, 3))
        sns.lineplot(x=grd.index, y=grd["count"], ax=ax1, color="#4C78A8")
        ax1.set_title("Daily transaction counts")
        ax1.set_xlabel("Date"); ax1.set_ylabel("Count")
        plt.tight_layout(); plt.show()

        if "__y__" in dft.columns:
            fig, ax2 = plt.subplots(figsize=(10, 3))
            sns.lineplot(x=grd.index, y=grd["fraud_rate"], ax=ax2, color="#F58518")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1%}"))
            ax2.set_title("Daily fraud rate")
            ax2.set_xlabel("Date"); ax2.set_ylabel("Fraud rate")
            plt.tight_layout(); plt.show()


def geolocation_analysis(df: pd.DataFrame, inferred: Dict[str, Optional[str]], plot_sample_n: int = 20000):
    lat, lon = inferred.get("latitude"), inferred.get("longitude")
    if not lat or not lon or lat not in df.columns or lon not in df.columns:
        print("Skipping geo analysis: latitude/longitude not found.")
        return
    dfg = df[[lat, lon]].copy()
    if "__y__" in df.columns:
        dfg["__y__"] = df["__y__"]
    dfg = dfg.dropna()
    dfg = sample_df(dfg, min(plot_sample_n, 50000))
    plt.figure(figsize=(6, 5))
    if "__y__" in dfg.columns:
        sns.scatterplot(data=dfg, x=lon, y=lat, hue="__y__", alpha=0.3, s=12, palette=["#4C78A8", "#F58518"])
        plt.legend(title="Fraud", loc="best")
    else:
        sns.scatterplot(data=dfg, x=lon, y=lat, alpha=0.3, s=12, color="#4C78A8")
    plt.title("Geospatial scatter (sample)")
    plt.tight_layout(); plt.show()


def device_and_user_analysis(df: pd.DataFrame, inferred: Dict[str, Optional[str]]):
    for col_key in ["device_type", "device_os", "merchant_category", "user_gender"]:
        col = inferred.get(col_key)
        if not col or col not in df.columns:
            continue
        tmp = df[[col]].copy()
        if "__y__" in df.columns:
            tmp["__y__"] = df["__y__"]
        vc = tmp[col].value_counts(dropna=False).head(20)
        plt.figure(figsize=(9, max(3, 0.3 * len(vc))))
        sns.barplot(x=vc.values, y=vc.index.astype(str), color="#4C78A8")
        plt.title(f"{col} distribution")
        plt.xlabel("Count"); plt.ylabel(col)
        plt.tight_layout(); plt.show()
        if "__y__" in tmp.columns:
            gr = tmp.groupby(col)["__y__"].mean().sort_values(ascending=False).head(20)
            plt.figure(figsize=(9, max(3, 0.3 * len(gr))))
            sns.barplot(x=gr.values, y=gr.index.astype(str), color="#F58518")
            plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1%}"))
            plt.title(f"Fraud rate by {col}")
            plt.xlabel("Fraud rate"); plt.ylabel(col)
            plt.tight_layout(); plt.show()


def duplicates_checks(df: pd.DataFrame, inferred: Dict[str, Optional[str]]):
    txid = inferred.get("transaction_id")
    if txid and txid in df.columns:
        dup_rate = float(df.duplicated(subset=[txid]).mean())
        print(f"Duplicate transaction_id rate: {dup_rate:.2%}")
    else:
        subset = [c for c in [inferred.get("user_id"), inferred.get("timestamp"), inferred.get("amount"), inferred.get("merchant_id")] if c in df.columns]
        if subset:
            dup_rate = float(df.duplicated(subset=subset).mean())
            print(f"Potential duplicate rows rate (by {subset}): {dup_rate:.2%}")
        else:
            print("Duplicate check: no suitable subset; skipped.")


# ----------------------------
# Orchestration
# ----------------------------

def run_all_eda(
    dataset_path: Optional[str] = None,
    column_hints: Optional[Dict[str, Optional[str]]] = None,
    plot_sample_n: int = 300000,
    auto_find_data: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Dict[str, List[str]]]:
    print("Loading dataset...")
    df, path = load_dataset(dataset_path, auto_find_data=auto_find_data)
    print(f"Loaded: {path}")
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")

    # Normalize column names (strip only)
    df.columns = [c.strip() for c in df.columns]

    print("\nInferring columns...")
    inferred = infer_columns(df, column_hints)
    for k, v in inferred.items():
        if v is not None:
            print(f"  {k:>18}: {v}")

    # Parse timestamp if needed
    ts_col = inferred.get("timestamp")
    if ts_col and not is_datetime64_any_dtype(df[ts_col]):
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        except Exception:
            pass

    # Build feature groups
    features = build_feature_groups(df, target_col=inferred.get("target"))
    display(pd.Series({k: len(v) for k, v in features.items()}).to_frame("count"))

    # EDA sections
    print("\n=== Overview ===")
    overview(df)

    print("\n=== Missingness ===")
    missingness(df)

    print("\n=== Class Imbalance ===")
    y, pos_label = class_imbalance(df, inferred.get("target"))

    print("\n=== Numeric Distributions & Outliers ===")
    numeric_distributions_and_outliers(df, inferred, features)

    print("\n=== Categorical Distributions & Fraud Rates ===")
    categorical_distributions(df, features)

    print("\n=== Correlations ===")
    correlation_analysis(df, features)

    print("\n=== Temporal Patterns ===")
    temporal_analysis(df, inferred)

    print("\n=== Geolocation (if available) ===")
    geolocation_analysis(df, inferred, plot_sample_n=min(plot_sample_n, 50000))

    print("\n=== Device/User/Merchant ===")
    device_and_user_analysis(df, inferred)

    print("\n=== Duplicates ===")
    duplicates_checks(df, inferred)

    print("\nEDA complete.")

    # Guidance
    print("\nNext-steps suggestions:")
    print("- Address missingness (impute/drop) per feature type and importance.")
    print("- Consider log-transforming highly skewed numeric features (e.g., amount).")
    print("- Encode categoricals (target encoding for high-cardinality; one-hot for low).")
    print("- Normalize/standardize numeric features as needed; cap/clip outliers.")
    print("- Time-based splits; beware of leakage when creating rolling/agg features.")
    print("- Rebalance strategies for modeling: class weights, undersampling/oversampling.")

    return df, inferred, features


__all__ = [
    "load_dataset",
    "infer_columns",
    "build_feature_groups",
    "run_all_eda",
    "overview",
    "missingness",
    "class_imbalance",
    "numeric_distributions_and_outliers",
    "categorical_distributions",
    "correlation_analysis",
    "temporal_analysis",
    "geolocation_analysis",
    "device_and_user_analysis",
    "duplicates_checks",
]
