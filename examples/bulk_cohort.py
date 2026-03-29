#!/usr/bin/env python3
"""
Replicate SecAct R vignette: bulkCohort
https://data2intelligence.github.io/SecAct/articles/bulkCohort.html

Infer secreted protein activity in a patient cohort and link to clinical outcomes.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lifelines import CoxPHFitter

from secactpy import secact_activity_inference
from secactpy.visualization import risk_lollipop

# --- Data path (from R SecAct package) ---
data_path = "/data/parks34/projects/0sigdiscov/pkg_dev/secact/inst/extdata/"

# --- 1. Prepare expression data ---
print("Loading expression data...")
expr = pd.read_csv(f"{data_path}Pancreatic_Nivolumab_Padron2022.logTPM.gz",
                    sep="\t", index_col=0)
print(f"Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")

# --- 2. Infer secreted protein activity ---
print("\nRunning SecAct inference...")
res = secact_activity_inference(
    input_profile=expr,
    input_profile_control=None,  # No control — normalize to zero mean
    verbose=True,
)

act = res["zscore"]
print(f"\nActivity matrix: {act.shape}")
print("\nActivity (first 6 proteins x 3 patients):")
print(act.iloc[:6, :3])

# --- 3. Load clinical data ---
print("\nLoading clinical data...")
clinical = pd.read_csv(f"{data_path}Pancreatic_Nivolumab_Padron2022.OS_Nivo+Sotiga+Chemo",
                        sep="\t", index_col=0)
print(f"Clinical data: {clinical.shape[0]} patients")
print(clinical.head())

# --- 4. Calculate clinical relevance (Cox PH regression) ---
print("\nComputing risk scores via Cox PH regression...")

# Overlap patients
common = act.columns.intersection(clinical.index)
print(f"Overlapping patients: {len(common)}")

act_common = act[common]
clin_common = clinical.loc[common]

# Cox regression for each protein
risk_scores = {}
try:
    cph = CoxPHFitter()
    for protein in act_common.index:
        try:
            df = clin_common[["Time", "Event"]].copy()
            df["activity"] = act_common.loc[protein].values.astype(float)
            df = df.dropna()
            if df["activity"].std() < 1e-10:
                continue
            cph.fit(df, duration_col="Time", event_col="Event", formula="activity")
            # Risk score = z-score (Coef / StdErr)
            coef = cph.summary_.loc["activity", "coef"]
            se = cph.summary_.loc["activity", "se(coef)"]
            risk_scores[protein] = coef / se if se > 0 else 0.0
        except Exception:
            continue
    print(f"Successfully computed risk scores for {len(risk_scores)} proteins")
except ImportError:
    # Fallback: Spearman correlation with survival time
    print("lifelines not available, using Spearman correlation as proxy...")
    for protein in act_common.index:
        vals = act_common.loc[protein, common].astype(float)
        surv = clin_common["Time"].astype(float)
        valid = vals.notna() & surv.notna()
        if valid.sum() > 5:
            r, _ = spearmanr(vals[valid], surv[valid])
            risk_scores[protein] = r

risk_series = pd.Series(risk_scores).sort_values()
print(f"\nRisk scores (top 6 high-risk):")
print(risk_series.tail(6))
print(f"\nRisk scores (top 6 low-risk):")
print(risk_series.head(6))

# --- 5. Visualize risk scores ---
print("\nGenerating risk score lollipop plot...")
fig = risk_lollipop(risk_series, title="Risk Score (Cox PH z-score)")
fig.write_html(f"{data_path}../../docs/bulk_cohort_python.html")
print(f"Plot saved to docs/bulk_cohort_python.html")

try:
    fig.write_image(f"{data_path}../../docs/bulk_cohort_python.png", width=800, height=600)
    print(f"Plot saved to docs/bulk_cohort_python.png")
except Exception:
    print("(Install kaleido for static image export)")

# --- 6. Survival curve for a top protein ---
print("\nGenerating survival curve for top risk protein...")
import plotly.graph_objects as go

top_protein = risk_series.index[-1]  # Highest risk
vals = act_common.loc[top_protein, common].astype(float)
median_val = vals.median()
high_group = vals[vals >= median_val].index
low_group = vals[vals < median_val].index

fig_surv = go.Figure()
for group, label, color in [(high_group, "High", "#e74c3c"), (low_group, "Low", "#3498db")]:
    times = clin_common.loc[group, "Time"].astype(float).sort_values()
    events = clin_common.loc[group, "Event"].astype(int).loc[times.index]
    n = len(times)
    # Kaplan-Meier estimator
    survival = []
    s = 1.0
    for i, (t, e) in enumerate(zip(times, events)):
        at_risk = n - i
        if e == 1 and at_risk > 0:
            s *= (at_risk - 1) / at_risk
        survival.append(s)
    fig_surv.add_trace(go.Scatter(
        x=times.values, y=survival,
        mode="lines", name=f"{label} activity (n={n})",
        line=dict(color=color, width=2),
    ))

fig_surv.update_layout(
    title=f"Survival: {top_protein}",
    xaxis_title="Overall Survival (Days)",
    yaxis_title="Survival Probability",
    template="plotly_white",
)
fig_surv.write_html(f"{data_path}../../docs/bulk_survival_python.html")
print(f"Survival plot saved to docs/bulk_survival_python.html")

print("\nDone! Replication of bulkCohort vignette complete.")
