"""
Downstream analysis functions for SecAct.

Post-inference analysis including survival analysis, signaling patterns,
and cell-cell communication. Mirrors R's SecAct/R/downstream.R.

When spatial-gpu is installed, delegates to its GPU-accelerated implementations.
Falls back to standalone implementations otherwise (bulk/SC users who don't
need the full spatial stack).
"""

from __future__ import annotations

import functools
import warnings
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
    "ccc_spatial",
]


# ---------------------------------------------------------------------------
# spatial-gpu availability check
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _has_spatialgpu() -> bool:
    try:
        import spatialgpu.deconvolution  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
# 1. Cox PH regression
# ===========================================================================
def coxph_regression(
    activity: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    """Cox proportional hazard regression linking activity to survival.

    For each protein, fits: Surv(Time, Event) ~ Activity + covariates.
    Risk scores are z-scores (Coef / StdErr) from the Wald test.

    Delegates to ``spatialgpu.deconvolution.secact_coxph_regression`` when
    spatial-gpu is installed. Falls back to a standalone lifelines implementation.

    Mirrors R's ``SecAct.coxph.regression(mat, surv)``.

    Parameters
    ----------
    activity : DataFrame
        Activity z-scores, proteins x samples.
    clinical : DataFrame
        Must have 'Time' and 'Event' columns. Additional columns are covariates.
        Index = sample IDs.

    Returns
    -------
    DataFrame
        Columns: 'risk_score_z', 'p_value'. Index = protein names.
    """
    if _has_spatialgpu():
        try:
            from spatialgpu.deconvolution.secact import secact_coxph_regression
            return secact_coxph_regression(activity, clinical)
        except Exception as e:
            warnings.warn(f"spatial-gpu delegation failed ({type(e).__name__}: {e}), using standalone")


    return _coxph_regression_standalone(activity, clinical)


def _coxph_regression_standalone(
    activity: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        raise ImportError(
            "lifelines is required for Cox regression.\n"
            "Install with: pip install lifelines\n"
            "Or install spatial-gpu for the GPU-accelerated version."
        )

    common = activity.columns.intersection(clinical.index)
    if len(common) < 5:
        raise ValueError(f"Only {len(common)} overlapping samples")

    act_common = activity[common]
    clin_common = clinical.loc[common].copy()

    time_col = _find_column(clin_common, ["Time", "time", "OS", "os", "survival_time"])
    event_col = _find_column(clin_common, ["Event", "event", "Status", "status", "OS_event"])
    if time_col is None or event_col is None:
        raise ValueError("Clinical data must have 'Time' and 'Event' columns")

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


# ===========================================================================
# 2. Signaling pattern (NMF)
# ===========================================================================
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

    Delegates to ``spatialgpu.deconvolution.secact_signaling_patterns`` when
    spatial-gpu is installed. Falls back to a standalone sklearn implementation.

    Mirrors R's ``SecAct.signaling.pattern()``.

    Parameters
    ----------
    activity : DataFrame
        Protein activity z-scores, proteins x spots.
    coordinates : DataFrame
        Spatial coordinates with 'x' and 'y' columns. Index = spot IDs.
    expression : DataFrame or sparse matrix
        Gene expression, genes x spots.
    k : int or list[int]
        Number of NMF factors. If list, selects optimal k.
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
        Keys: 'ccc_sp', 'weight_W', 'signal_H', 'k'.
    """
    if _has_spatialgpu():
        try:
            from spatialgpu.deconvolution.secact import secact_signaling_patterns
            import anndata as ad
            # Build AnnData: spatial-gpu expects spots x genes orientation
            if isinstance(expression, pd.DataFrame):
                X = expression.T.values  # genes x spots → spots x genes
                var = pd.DataFrame(index=expression.index)
            elif sparse.issparse(expression):
                X = expression.T  # transpose sparse
                var = pd.DataFrame(index=range(expression.shape[0]))
            else:
                X = expression.T
                var = pd.DataFrame(index=range(expression.shape[0]))

            obs = pd.DataFrame(index=coordinates.index)
            adata = ad.AnnData(X=X, obs=obs, var=var)
            adata.obsm["spatial"] = coordinates[["x", "y"]].values

            secact_signaling_patterns(adata, k=k, radius=radius,
                                      scale_factor=scale_factor)
            secact_out = adata.uns.get("spacet", {}).get("SecAct_output", {})
            pattern = secact_out.get("pattern", {})
            return {
                "ccc_sp": pattern.get("ccc_sp", pd.DataFrame()),
                "weight_W": pattern.get("weight_W"),
                "signal_H": pattern.get("signal_H"),
                "k": k if isinstance(k, int) else pattern.get("k", 0),
            }
        except Exception as e:
            warnings.warn(f"spatial-gpu delegation failed ({type(e).__name__}: {e}), using standalone")


    return _signaling_pattern_standalone(activity, coordinates, expression, k,
                                         radius, scale_factor, sigma, corr_p_cutoff)


def _signaling_pattern_standalone(
    activity, coordinates, expression, k,
    radius, scale_factor, sigma, corr_p_cutoff,
) -> dict:
    from sklearn.decomposition import NMF as NMFModel
    from scipy.spatial import cKDTree

    common_spots = activity.columns.intersection(coordinates.index)
    act = activity[common_spots]
    coords = coordinates.loc[common_spots, ["x", "y"]].values
    n = len(coords)

    # Gaussian spatial weights via KDTree (O(n log n) instead of O(n²))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    weights = sparse.lil_matrix((n, n))
    for i, j in pairs:
        d = np.linalg.norm(coords[i] - coords[j])
        w = np.exp(-d ** 2 / (2 * sigma ** 2))
        weights[i, j] = w
        weights[j, i] = w
    weights = weights.tocsr()
    # Row-normalize
    row_sums = np.array(weights.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    weights = sparse.diags(1 / row_sums) @ weights

    # Normalize expression
    if isinstance(expression, pd.DataFrame):
        expr_dense = expression[common_spots].values
        gene_names = expression.index
    elif sparse.issparse(expression):
        expr_dense = np.asarray(expression.todense())
        gene_names = [f"gene_{i}" for i in range(expression.shape[0])]
    else:
        expr_dense = expression
        gene_names = [f"gene_{i}" for i in range(expression.shape[0])]

    col_sums = expr_dense.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    expr_norm = np.log2(expr_dense / col_sums * scale_factor + 1)

    # Aggregate neighbor expression
    expr_aggr = expr_norm @ weights.T

    # Correlate activity with aggregated neighbor expression
    corr_results = {}
    for protein in act.index:
        if protein in gene_names:
            gene_idx = list(gene_names).index(protein)
            r, p = spearmanr(act.loc[protein].values.astype(float), expr_aggr[gene_idx, :])
            corr_results[protein] = {"r": r, "p": p}

    corr_df = pd.DataFrame(corr_results).T
    if corr_df.empty:
        return {"ccc_sp": corr_df, "weight_W": None, "signal_H": None, "k": 0}

    corr_genes = corr_df[corr_df["p"] < corr_p_cutoff].index.tolist()
    if len(corr_genes) < 2:
        return {"ccc_sp": corr_df, "weight_W": None, "signal_H": None, "k": 0}

    # NMF
    act_filtered = act.loc[corr_genes].values.astype(float)
    act_filtered[act_filtered < 0] = 0

    if isinstance(k, (list, tuple)):
        best_k, best_err = k[0], np.inf
        for ki in k:
            model = NMFModel(n_components=ki, random_state=123456, max_iter=500)
            model.fit_transform(act_filtered)
            if model.reconstruction_err_ < best_err:
                best_err = model.reconstruction_err_
                best_k = ki
        k_final = best_k
    else:
        k_final = k

    model = NMFModel(n_components=k_final, random_state=123456, max_iter=500)
    W = model.fit_transform(act_filtered)
    H = model.components_

    factor_names = [str(i + 1) for i in range(k_final)]
    weight_W = pd.DataFrame(W, index=corr_genes, columns=factor_names)
    signal_H = pd.DataFrame(H, index=factor_names, columns=common_spots)

    return {"ccc_sp": corr_df, "weight_W": weight_W, "signal_H": signal_H, "k": k_final}


# ===========================================================================
# 3. Pattern-associated genes
# ===========================================================================
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
        Subset of W where pattern n has the highest weight, sorted descending.
    """
    col = str(n)
    if col not in weight_W.columns:
        raise ValueError(f"Pattern {n} not found. Available: {list(weight_W.columns)}")

    max_pattern = weight_W.idxmax(axis=1)
    return weight_W.loc[max_pattern == col].sort_values(col, ascending=False)


# ===========================================================================
# 4. CCC from scRNA-seq (unique to SecActpy — no spatial-gpu equivalent)
# ===========================================================================
def ccc_scrnaseq(
    adata,
    cell_type_col: str,
    condition_col: Optional[str] = None,
    case: Optional[str] = None,
    control: Optional[str] = None,
    act_diff_cutoff: float = 2.0,
    exp_logfc_cutoff: float = 0.2,
    exp_frac_cutoff: float = 0.1,
    sig_matrix: str = "secact",
) -> pd.DataFrame:
    """Infer cell-cell communication from scRNA-seq data.

    For each cell type, computes activity change (case vs control) and expression
    change, then matches senders (high expression) to receivers (high activity)
    using a ligand-receptor database.

    Unique to SecActpy — mirrors R's ``SecAct.CCC.scRNAseq()``.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with cell type and condition annotations.
    cell_type_col : str
        Column in adata.obs with cell type labels.
    condition_col : str, optional
        Column for condition labels.
    case, control : str, optional
        Condition labels.
    act_diff_cutoff : float
        Activity z-score cutoff.
    exp_logfc_cutoff : float
        Log fold change cutoff.
    exp_frac_cutoff : float
        Fraction expressing cutoff.
    sig_matrix : str
        Signature matrix name.

    Returns
    -------
    DataFrame
        CCC interactions: sender, receiver, secretedProtein, activity_z.
    """
    from secactpy import secact_activity_inference

    cell_types = list(adata.obs[cell_type_col].unique())

    if condition_col is not None:
        case_types = set(adata.obs.loc[adata.obs[condition_col] == case, cell_type_col])
        ctrl_types = set(adata.obs.loc[adata.obs[condition_col] == control, cell_type_col])
        cell_types = list(case_types & ctrl_types)

    lr_db = _load_lr_database()
    if lr_db.empty:
        warnings.warn("LR database not loaded (spatial-gpu not installed?). CCC results will be empty.")
        return pd.DataFrame(columns=["sender", "receiver", "secretedProtein", "activity_z"])
    all_results = []

    for ct in cell_types:
        if condition_col is not None:
            mask_case = (adata.obs[cell_type_col] == ct) & (adata.obs[condition_col] == case)
            mask_ctrl = (adata.obs[cell_type_col] == ct) & (adata.obs[condition_col] == control)
        else:
            mask_case = adata.obs[cell_type_col] == ct
            mask_ctrl = adata.obs[cell_type_col] != ct

        adata_case = adata[mask_case]
        adata_ctrl = adata[mask_ctrl]
        if adata_case.n_obs < 3 or adata_ctrl.n_obs < 3:
            continue

        # Pseudo-bulk expression
        expr_case = _pseudo_bulk_mean(adata_case)
        expr_ctrl = _pseudo_bulk_mean(adata_ctrl)

        logfc = np.log2(expr_case + 1) - np.log2(expr_ctrl + 1)
        logfc_series = pd.Series(logfc, index=adata.var_names)

        frac = np.asarray((adata_case.X > 0).mean(axis=0)).ravel() if sparse.issparse(adata_case.X) else (adata_case.X > 0).mean(axis=0)
        frac_series = pd.Series(frac, index=adata.var_names)

        # Activity inference
        try:
            pseudo_case = pd.DataFrame(expr_case, index=adata.var_names, columns=["case"])
            pseudo_ctrl = pd.DataFrame(expr_ctrl, index=adata.var_names, columns=["control"])
            res = secact_activity_inference(pseudo_case, input_profile_control=pseudo_ctrl, verbose=False)
            act_change = res.get("zscore", pd.DataFrame())
            if act_change.empty:
                continue
            act_series = act_change.iloc[:, 0] if act_change.ndim > 1 else act_change
        except Exception:
            continue

        receiver_proteins = act_series[act_series.abs() > act_diff_cutoff].index.tolist()
        sender_genes = logfc_series[(logfc_series.abs() > exp_logfc_cutoff) & (frac_series > exp_frac_cutoff)].index.tolist()

        for protein in receiver_proteins:
            if protein in sender_genes:
                for sender_ct in cell_types:
                    if sender_ct == ct:
                        continue
                    all_results.append({
                        "sender": sender_ct,
                        "receiver": ct,
                        "secretedProtein": protein,
                        "activity_z": float(act_series.get(protein, 0)),
                    })

    if not all_results:
        return pd.DataFrame(columns=["sender", "receiver", "secretedProtein", "activity_z"])
    return pd.DataFrame(all_results)


# ===========================================================================
# 5. Spatial CCC — delegates to spatial-gpu (no standalone fallback)
# ===========================================================================
def ccc_spatial(
    adata,
    cell_type_col: str,
    radius: float = 20.0,
    scale_factor: float = 1000.0,
    ratio_cutoff: float = 0.2,
    padj_cutoff: float = 0.01,
    n_background: int = 1000,
    seed: int = 123,
    n_jobs: int = 1,
):
    """Compute spatial cell-cell communication mediated by secreted proteins.

    Requires spatial-gpu. Mirrors R's ``SecAct.CCC.scST()``.

    For each cell-type pair, tests whether neighboring cells communicate
    via secreted proteins (expression x activity > 0) more than expected
    by a permutation background.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct activity results (run secact_inference first)
        and cell type annotations.
    cell_type_col : str
        Column in adata.obs with cell type labels.
    radius : float
        Neighbor radius in coordinate units.
    scale_factor : float
        TPM normalization scale factor.
    ratio_cutoff : float
        Minimum ratio of communicating pairs.
    padj_cutoff : float
        BH-adjusted p-value cutoff.
    n_background : int
        Number of background permutations.
    seed : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    AnnData with CCC results stored in uns.
    """
    if not _has_spatialgpu():
        raise ImportError(
            "spatial-gpu is required for spatial CCC analysis.\n"
            "Install with: pip install spatial-gpu\n"
            "For non-spatial CCC, use ccc_scrnaseq() instead."
        )

    from spatialgpu.deconvolution.secact import secact_spatial_ccc
    return secact_spatial_ccc(
        adata,
        cell_type_col=cell_type_col,
        scale_factor=scale_factor,
        radius=radius,
        ratio_cutoff=ratio_cutoff,
        padj_cutoff=padj_cutoff,
        n_background=n_background,
        seed=seed,
        n_jobs=n_jobs,
    )


# ===========================================================================
# Helpers
# ===========================================================================
def _pseudo_bulk_mean(adata) -> np.ndarray:
    """Compute mean expression per gene across cells, handling sparse matrices."""
    if sparse.issparse(adata.X):
        return np.asarray(adata.X.mean(axis=0)).ravel()
    return np.asarray(adata.X.mean(axis=0)).ravel()


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    for c in candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            return matches[0]
    return None


@functools.lru_cache(maxsize=1)
def _load_lr_database() -> pd.Series:
    """Load ligand-receptor interaction database. Cached after first call."""
    import importlib.resources

    try:
        import spatialgpu
        lr_path = importlib.resources.files(spatialgpu) / "data" / "Ramilowski2015.txt"
        lr = pd.read_csv(str(lr_path), sep="\t")
        return lr.set_index("Ligand.ApprovedSymbol")["Receptor.ApprovedSymbol"]
    except (ImportError, FileNotFoundError, KeyError):
        pass

    return pd.Series(dtype=str)
