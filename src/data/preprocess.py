import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    def __init__(self, schema: Dict, seed: int = 42):
        self.schema = schema
        self.seed = seed
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = RobustScaler()

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.schema['id_col']
        if id_col in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[id_col])
        else:
            before = len(df)
            df = df.drop_duplicates()
        return df

    def cap_outliers(self, df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
        # IQR capping for numeric columns
        for col in self.schema['num_cols']:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Impute
        df[self.schema['num_cols']] = self.num_imputer.fit_transform(df[self.schema['num_cols']])
        df[self.schema['cat_cols']] = self.cat_imputer.fit_transform(df[self.schema['cat_cols']])
        # Scale only amount-like and trust scores
        scale_cols = [c for c in self.schema['num_cols']]
        df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.schema['num_cols']] = self.num_imputer.transform(df[self.schema['num_cols']])
        df[self.schema['cat_cols']] = self.cat_imputer.transform(df[self.schema['cat_cols']])
        scale_cols = [c for c in self.schema['num_cols']]
        df[scale_cols] = self.scaler.transform(df[scale_cols])
        return df


def temporal_split(df: pd.DataFrame, schema: Dict, train_frac: float, val_frac: float, test_frac: float):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    df = df.sort_values(schema['timestamp_col'])
    n = len(df)
    i_train = int(train_frac * n)
    i_val = int((train_frac + val_frac) * n)
    train = df.iloc[:i_train]
    val = df.iloc[i_train:i_val]
    test = df.iloc[i_val:]
    return train, val, test
