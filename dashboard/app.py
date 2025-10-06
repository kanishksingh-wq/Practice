import json
import os
import pandas as pd
import streamlit as st
import plotly.express as px

ART_DIR = 'artifacts'

st.set_page_config(page_title='Fraud Detection Dashboard', layout='wide')
st.title('Fraud Detection: Results Overview')

metrics_path = os.path.join(ART_DIR, 'metrics.json')
psi_path = os.path.join(ART_DIR, 'psi.json')

if not os.path.exists(metrics_path):
    st.warning('No metrics.json found in artifacts/. Run main.py first.')
    st.stop()

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

cols = st.columns(2)
with cols[0]:
    st.subheader('Model comparison (Test AUC)')
    auc_df = pd.DataFrame([{'model': k, 'test_auc': v['test_auc'], 'val_auc': v['val_auc']} for k, v in metrics.items()])
    st.dataframe(auc_df.sort_values('test_auc', ascending=False), use_container_width=True)

with cols[1]:
    st.subheader('Threshold metrics (Test)')
    th_df = pd.DataFrame([{**{'model': k}, **{kk: vv for kk, vv in v.items() if kk not in ['tn','fp','fn','tp']}} for k, v in metrics.items()])
    st.dataframe(th_df, use_container_width=True)

st.subheader('Confusion matrices')
for k, v in metrics.items():
    cm_df = pd.DataFrame({'TN': [v['tn']], 'FP': [v['fp']], 'FN': [v['fn']], 'TP': [v['tp']]})
    st.write(f'Model: {k}')
    st.dataframe(cm_df)

if os.path.exists(psi_path):
    st.subheader('PSI (Feature Drift)')
    with open(psi_path, 'r') as f:
        psi_scores = json.load(f)
    psi_df = pd.DataFrame({'feature': list(psi_scores.keys()), 'psi': list(psi_scores.values())})
    fig = px.bar(psi_df.sort_values('psi', ascending=False).head(30), x='feature', y='psi')
    st.plotly_chart(fig, use_container_width=True)

# SHAP images
st.subheader('Explainability (SHAP)')
images = [f for f in os.listdir(ART_DIR) if f.endswith('.png')]
if images:
    for img in images:
        st.image(os.path.join(ART_DIR, img), caption=img, use_column_width=True)
else:
    st.info('No SHAP images found yet.')
