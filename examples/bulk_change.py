#!/usr/bin/env python3
"""
Replicate SecAct R vignette: bulkChange
https://data2intelligence.github.io/SecAct/articles/bulkChange.html

Infer secreted protein activity change between treatment and control (anti-IFNG vs control).
"""

import pandas as pd
from secactpy import secact_activity_inference
from secactpy.visualization import activity_change_bar

# --- Data path (from R SecAct package) ---
data_path = "/data/parks34/projects/0sigdiscov/pkg_dev/secact/inst/extdata/"

# --- 1. Prepare expression data ---
print("Loading expression data...")
expr = pd.read_csv(f"{data_path}GSE100093.IFNG.expr.gz", sep="\t", index_col=0, compression=None)
meta = pd.read_csv(f"{data_path}GSE100093.IFNG.meta", sep="\t", index_col=0)

print(f"Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")
print(f"Metadata: {meta.shape[0]} samples")

# Subset treatment and control groups
treatment_samples = meta.index[meta["Treatment"] == "Anti-IFNG"]
control_samples = meta.index[meta["Treatment"] == "Control"]

expr_treatment = expr[treatment_samples]
expr_control = expr[control_samples]

print(f"Treatment: {expr_treatment.shape[1]} samples")
print(f"Control: {expr_control.shape[1]} samples")

# --- 2. Infer activity change ---
print("\nRunning SecAct inference (treatment vs control)...")
res = secact_activity_inference(
    input_profile=expr_treatment,
    input_profile_control=expr_control,
    is_single_sample_level=False,
    verbose=True,
)

act_change = res["zscore"]
print(f"\nActivity change matrix: {act_change.shape}")
print("\nTop 6 activity changes:")
print(act_change.head(6))

# --- 3. Check IFNG activity ---
if "IFNG" in act_change.index:
    ifng_change = act_change.loc["IFNG"].values[0] if act_change.ndim > 1 else act_change.loc["IFNG"]
    print(f"\nIFNG activity change: {ifng_change:.4f}")
    print("(Expected: negative, consistent with anti-IFNG treatment)")
else:
    print("\nIFNG not found in results")

# --- 4. Visualize activity change ---
print("\nGenerating activity change bar plot...")

# Get z-score series
if isinstance(act_change, pd.DataFrame):
    if act_change.shape[1] == 1:
        zscore_series = act_change.iloc[:, 0]
    else:
        zscore_series = act_change.mean(axis=1)
else:
    zscore_series = act_change

fig = activity_change_bar(zscore_series, title="Activity Change (Anti-IFNG vs Control)")
fig.write_html(f"{data_path}../../docs/bulk_change_python.html")
print(f"Plot saved to docs/bulk_change_python.html")

# Also save as static image if kaleido available
try:
    fig.write_image(f"{data_path}../../docs/bulk_change_python.png", width=800, height=500)
    print(f"Plot saved to docs/bulk_change_python.png")
except Exception:
    print("(Install kaleido for static image export: pip install kaleido)")

print("\nDone! Replication of bulkChange vignette complete.")
