import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from typing import List


def select_by_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
    mi = mutual_info_classif(X.fillna(0), y, discrete_features='auto', random_state=42)
    order = np.argsort(mi)[::-1]
    k = min(k, X.shape[1])
    selected = X.columns[order][:k].tolist()
    return selected
