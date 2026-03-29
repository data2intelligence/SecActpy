"""
Downstream analysis functions for SecAct.

Post-inference analysis including survival analysis, signaling patterns,
and cell-cell communication. Mirrors R's SecAct/R/downstream.R.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

__all__ = [
    "coxph_regression",
    "signaling_pattern",
    "signaling_pattern_gene",
    "ccc_scrnaseq",
]


def coxph_regression(
    activity: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    """Cox proportional hazard regression linking activity to survival.

    For each protein, fits: Surv(Time, Event) ~ Activity + covariates.
    Risk scores are z-scores (Coef / StdErr) from the Wald test.

    Mirrors R's ``SecAct.coxph.regression(mat, surv)``.

    Parameters
    ----------
    activity : DataFrame
        Activity z-scores, proteins x samples.
    clinical : DataFrame
        Must have 'Time' and 'Event' columns. Additional columns are used as covariates.
        Index = sample IDs.

    Returns
    -------
    DataFrame
        Columns: 'risk_score_z', 'p_value'. Index = protein names.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        raise ImportError(
            "lifelines is required for Cox regression.\n"
            "Install with: pip install lifelines"
        )

    # Align samples
    common = activity.columns.intersection(clinical.index)
    if len(common) < 5:
        raise ValueError(f"Only {len(common)} overlapping samples between activity and clinical data")

    act_common = activity[common]
    clin_common = clinical.loc[common].copy()

    # Ensure Time and Event columns exist
    time_col = _find_column(clin_common, ["Time", "time", "OS", "os", "survival_time"])
    event_col = _find_column(clin_common, ["Event", "event", "Status", "status", "OS_event"])
    if time_col is None or event_col is None:
        raise ValueError("Clinical data must have 'Time' and 'Event' columns")

    # Identify covariate columns (everything except Time and Event)
    covariate_cols = [c for c in clin_common.columns if c not in [time_col, event_col]]

    results = {}
    cph = CoxPHFitter()

    for protein in act_common.index:
        try:
            df = clin_common[[time_col, event_col] + covariate_cols].copy()
            df["Act"] = act_common.loc[protein].astype(float).values
            df = df.dropna()

            if df["Act"].std() < 1e-10 or len(df) < 5:
                continue

            cph.fit(df, duration_col=time_col, event_col=event_col)

            coef = cph.summary_.loc["Act", "coef"]
            se = cph.summary_.loc["Act", "se(coef)"]
            p = cph.summary_.loc["Act", "p"]
            z = coef / se if se > 0 else 0.0

            results[protein] = {"risk_score_z": z, "p_value": p}
        except Exception:
            continue

    return pd.DataFrame(results).T


def signaling_pattern(
    activity: pd.DataFrame,
    coordinates: pd.DataFrame,
    expression: Union[pd.DataFrame, sparse.spmatrix],
    k: Union[int, list[int]],
    radius: float = 200,
    scale_factor: float = 1e5,
    sigma: float = 100,
    corr_p_cutoff: float = 0.05,
) -> dict:
    """Discover signaling patterns via NMF on spatially-filtered activity.

    Mirrors R's ``SecAct.signaling.pattern()``.

    Steps:
    1. Compute Gaussian spatial weights within radius
    2. Aggregate neighbor expression using weights
    3. Correlate each protein's activity with its neighbor expression
    4. Filter proteins with significant spatial correlation
    5. Run NMF on non-negative activity of filtered proteins

    Parameters
    ----------
    activity : DataFrame
        Protein activity z-scores, proteins x spots.
    coordinates : DataFrame
        Spatial coordinates with columns 'x' and 'y'. Index = spot IDs.
    expression : DataFrame or sparse matrix
        Gene expression, genes x spots. Must include secreted protein genes.
    k : int or list[int]
        Number of NMF factors. If list, selects optimal k by silhouette.
    radius : float
        Spatial radius for neighbor weighting (micrometers).
    scale_factor : float
        Normalization scale factor.
    sigma : float
        Gaussian kernel bandwidth.
    corr_p_cutoff : float
        P-value cutoff for filtering correlated proteins.

    Returns
    -------
    dict
        Keys: 'ccc_sp' (DataFrame of correlations), 'weight_W' (NMF W matrix),
        'signal_H' (NMF H matrix), 'k' (selected k).
    """
    from sklearn.decomposition import NMF as NMFModel

    # Align spots
    common_spots = activity.columns.intersection(coordinates.index)
    act = activity[common_spots]
    coords = coordinates.loc[common_spots, ["x", "y"]].values

    # Step 1: Gaussian spatial weights
    print("Step 1. Computing spatial weights...")
    from scipy.spatial.distance import cdist
    dist_mat = cdist(coords, coords)
    weights = np.exp(-dist_mat ** 2 / (2 * sigma ** 2))
    weights[dist_mat > radius] = 0
    np.fill_diagonal(weights, 0)
    # Row-normalize
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums

    # Normalize expression to log-TPM
    if sparse.issparse(expression):
        expr = expression[:, [list(expression.columns if hasattr(expression, 'columns') else range(expression.shape[1])).index(s) for s in common_spots]]
        expr_dense = np.asarray(expr.todense())
    elif isinstance(expression, pd.DataFrame):
        expr_dense = expression[common_spots].values
    else:
        expr_dense = expression

    col_sums = expr_dense.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    expr_norm = expr_dense / col_sums * scale_factor
    expr_norm = np.log2(expr_norm + 1)

    # Aggregate neighbor expression
    expr_aggr = expr_norm @ weights.T  # genes x spots

    # Get gene names
    if isinstance(expression, pd.DataFrame):
        gene_names = expression.index
    else:
        gene_names = [f"gene_{i}" for i in range(expression.shape[0])]

    # Step 2: Correlate activity with aggregated neighbor expression
    print("Step 2. Computing spatial correlations...")
    corr_results = {}
    for protein in act.index:
        if protein in gene_names:
            gene_idx = list(gene_names).index(protein)
            act_vals = act.loc[protein].values.astype(float)
            expr_vals = expr_aggr[gene_idx, :]
            r, p = spearmanr(act_vals, expr_vals)
            corr_results[protein] = {"r": r, "p": p}

    corr_df = pd.DataFrame(corr_results).T
    if corr_df.empty:
        return {"ccc_sp": corr_df, "weight_W": None, "signal_H": None, "k": 0}

    # Filter significant correlations
    corr_genes = corr_df[corr_df["p"] < corr_p_cutoff].index.tolist()
    if len(corr_genes) < 2:
        print(f"Only {len(corr_genes)} genes pass correlation filter")
        return {"ccc_sp": corr_df, "weight_W": None, "signal_H": None, "k": 0}

    print(f"  {len(corr_genes)} proteins pass spatial correlation filter")

    # Step 3: NMF on non-negative activity
    print("Step 3. Running NMF...")
    act_filtered = act.loc[corr_genes].values.astype(float)
    act_filtered[act_filtered < 0] = 0

    if isinstance(k, (list, tuple)):
        # Select optimal k by reconstruction error
        best_k = k[0]
        best_err = np.inf
        for ki in k:
            model = NMFModel(n_components=ki, random_state=123456, max_iter=500)
            W = model.fit_transform(act_filtered)
            err = model.reconstruction_err_
            if err < best_err:
                best_err = err
                best_k = ki
        k_final = best_k
        print(f"  Optimal k = {k_final}")
    else:
        k_final = k

    model = NMFModel(n_components=k_final, random_state=123456, max_iter=500)
    W = model.fit_transform(act_filtered)
    H = model.components_

    weight_W = pd.DataFrame(W, index=corr_genes,
                            columns=[str(i + 1) for i in range(k_final)])
    signal_H = pd.DataFrame(H, index=[str(i + 1) for i in range(k_final)],
                            columns=common_spots)

    return {"ccc_sp": corr_df, "weight_W": weight_W, "signal_H": signal_H, "k": k_final}


def signaling_pattern_gene(weight_W: pd.DataFrame, n: int) -> pd.DataFrame:
    """Extract proteins associated with signaling pattern n.

    Mirrors R's ``SecAct.signaling.pattern.gene()``.

    Parameters
    ----------
    weight_W : DataFrame
        NMF weight matrix W from ``signaling_pattern()``. Proteins x factors.
    n : int
        Pattern number (1-indexed).

    Returns
    -------
    DataFrame
        Subset of W where pattern n has the highest weight, sorted by weight descending.
    """
    col = str(n)
    if col not in weight_W.columns:
        raise ValueError(f"Pattern {n} not found. Available: {list(weight_W.columns)}")

    # Proteins where pattern n has the maximum weight
    max_pattern = weight_W.idxmax(axis=1)
    mask = max_pattern == col
    result = weight_W.loc[mask].sort_values(col, ascending=False)
    return result


def ccc_scrnaseq(
    adata,
    cell_type_col: str,
    condition_col: Optional[str] = None,
    case: Optional[str] = None,
    control: Optional[str] = None,
    act_diff_cutoff: float = 2.0,
    exp_logfc_cutoff: float = 0.2,
    exp_mean_cutoff: float = 2.0,
    exp_frac_cutoff: float = 0.1,
    padj_cutoff: float = 0.01,
    sig_matrix: str = "secact",
    scale_factor: float = 1e5,
    lambda_: float = 5e5,
    n_rand: int = 1000,
) -> pd.DataFrame:
    """Infer cell-cell communication from scRNA-seq data.

    For each cell type, computes activity change (case vs control) and expression
    change, then matches senders (high expression) to receivers (high activity)
    using a ligand-receptor database.

    Mirrors R's ``SecAct.CCC.scRNAseq()``.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with cell type and condition annotations.
    cell_type_col : str
        Column in adata.obs with cell type labels.
    condition_col : str, optional
        Column in adata.obs with condition labels (e.g., 'treatment').
    case : str, optional
        Case condition label.
    control : str, optional
        Control condition label.
    act_diff_cutoff : float
        Activity z-score cutoff for significant change.
    exp_logfc_cutoff : float
        Log fold change cutoff for expression change.
    exp_mean_cutoff : float
        Mean expression cutoff.
    exp_frac_cutoff : float
        Fraction of cells expressing cutoff.
    padj_cutoff : float
        Adjusted p-value cutoff.
    sig_matrix : str
        Signature matrix name ('secact' or 'cytosig').
    scale_factor : float
        Normalization scale factor.
    lambda_ : float
        Ridge regression penalty.
    n_rand : int
        Number of permutations.

    Returns
    -------
    DataFrame
        CCC interactions with columns: sender, receiver, secretedProtein.
    """
    import anndata as ad
    from secactpy import secact_activity_inference, load_signature

    # Load signature to get list of secreted proteins
    sig = load_signature(sig_matrix)
    sp_genes = sig.columns.tolist()

    # Get cell types
    cell_types = adata.obs[cell_type_col].unique()
    if condition_col is not None:
        # Only cell types present in both conditions
        case_types = set(adata.obs.loc[adata.obs[condition_col] == case, cell_type_col])
        ctrl_types = set(adata.obs.loc[adata.obs[condition_col] == control, cell_type_col])
        cell_types = list(case_types & ctrl_types)

    print(f"Analyzing {len(cell_types)} cell types...")

    # Load LR database
    lr_db = _load_lr_database()

    # For each cell type, compute activity and expression changes
    all_results = []

    for ct in cell_types:
        print(f"  Processing {ct}...")

        if condition_col is not None:
            mask_case = (adata.obs[cell_type_col] == ct) & (adata.obs[condition_col] == case)
            mask_ctrl = (adata.obs[cell_type_col] == ct) & (adata.obs[condition_col] == control)
        else:
            mask_case = adata.obs[cell_type_col] == ct
            mask_ctrl = adata.obs[cell_type_col] != ct  # rest as control

        adata_case = adata[mask_case]
        adata_ctrl = adata[mask_ctrl]

        if adata_case.n_obs < 3 or adata_ctrl.n_obs < 3:
            continue

        # Pseudo-bulk expression
        if sparse.issparse(adata_case.X):
            expr_case = pd.DataFrame(
                np.asarray(adata_case.X.mean(axis=0)).ravel(),
                index=adata_case.var_names,
            )
            expr_ctrl = pd.DataFrame(
                np.asarray(adata_ctrl.X.mean(axis=0)).ravel(),
                index=adata_ctrl.var_names,
            )
        else:
            expr_case = pd.DataFrame(adata_case.X.mean(axis=0), index=adata_case.var_names)
            expr_ctrl = pd.DataFrame(adata_ctrl.X.mean(axis=0), index=adata_ctrl.var_names)

        # Expression log fold change
        logfc = np.log2(expr_case.values.ravel() + 1) - np.log2(expr_ctrl.values.ravel() + 1)
        logfc_series = pd.Series(logfc, index=adata_case.var_names)

        # Fraction expressing in case
        if sparse.issparse(adata_case.X):
            frac = np.asarray((adata_case.X > 0).mean(axis=0)).ravel()
        else:
            frac = (adata_case.X > 0).mean(axis=0)
        frac_series = pd.Series(frac, index=adata_case.var_names)

        # Activity inference (pseudo-bulk)
        try:
            pseudo_case = pd.DataFrame(
                np.asarray(adata_case.X.mean(axis=0)).ravel() if sparse.issparse(adata_case.X)
                else adata_case.X.mean(axis=0),
                index=adata_case.var_names, columns=["case"],
            )
            pseudo_ctrl = pd.DataFrame(
                np.asarray(adata_ctrl.X.mean(axis=0)).ravel() if sparse.issparse(adata_ctrl.X)
                else adata_ctrl.X.mean(axis=0),
                index=adata_ctrl.var_names, columns=["control"],
            )

            res = secact_activity_inference(
                pseudo_case, input_profile_control=pseudo_ctrl, verbose=False,
            )
            act_change = res.get("zscore", pd.DataFrame())
            if act_change.empty:
                continue
            act_series = act_change.iloc[:, 0] if act_change.ndim > 1 else act_change
        except Exception:
            continue

        # Filter: high activity change proteins for this cell type (receiver)
        receiver_proteins = act_series[act_series.abs() > act_diff_cutoff].index.tolist()

        # Filter: high expression change genes (sender)
        sender_mask = (
            (logfc_series.abs() > exp_logfc_cutoff) &
            (frac_series > exp_frac_cutoff)
        )
        sender_genes = logfc_series[sender_mask].index.tolist()

        # Match via LR database
        for protein in receiver_proteins:
            if protein not in lr_db.index:
                continue
            receptors = lr_db.loc[protein]
            if isinstance(receptors, str):
                receptors = [receptors]
            elif isinstance(receptors, pd.Series):
                receptors = receptors.values.tolist()

            # Check if ligand (= secreted protein gene) is expressed by any sender cell type
            for sender_ct in cell_types:
                if sender_ct == ct:
                    continue
                # Check if protein gene is upregulated in sender
                if protein in sender_genes:
                    all_results.append({
                        "sender": sender_ct,
                        "receiver": ct,
                        "secretedProtein": protein,
                        "activity_z": act_series.get(protein, 0),
                    })

    if not all_results:
        return pd.DataFrame(columns=["sender", "receiver", "secretedProtein", "activity_z"])

    return pd.DataFrame(all_results)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    for c in candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            return matches[0]
    return None


def _load_lr_database() -> pd.Series:
    """Load ligand-receptor interaction database.

    Uses the Ramilowski 2015 database bundled with spatial-gpu or SecAct.
    Returns a Series: ligand -> receptor.
    """
    import importlib.resources

    # Try spatial-gpu first
    try:
        import spatialgpu
        lr_path = importlib.resources.files(spatialgpu) / "data" / "Ramilowski2015.txt"
        lr = pd.read_csv(str(lr_path), sep="\t")
        return lr.set_index("Ligand.ApprovedSymbol")["Receptor.ApprovedSymbol"]
    except (ImportError, FileNotFoundError, KeyError):
        pass

    # Fallback: return empty
    return pd.Series(dtype=str)
