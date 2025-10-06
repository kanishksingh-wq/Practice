import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional


def shap_global_plots(model, X_train: pd.DataFrame, X_sample: pd.DataFrame, out_dir: str, prefix: str = 'model'):
    os.makedirs(out_dir, exist_ok=True)
    try:
        explainer = shap.Explainer(model, X_train)
    except Exception:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title('SHAP Beeswarm')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_shap_beeswarm.png'), dpi=150)
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title('SHAP Feature Importance (bar)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{prefix}_shap_bar.png'), dpi=150)
    plt.close()
