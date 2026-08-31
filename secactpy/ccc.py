"""Cell-cell communication (CCC) inference for SecActpy.

Python port of R SecAct's ``SecAct.CCC.scRNAseq`` (the scCCC vignette).

CCC is identified by two coupled criteria, optionally between two conditions
(e.g. Metastatic vs Primary):

1. up-regulated **expression** of a secreted protein in a *sender* cell type, and
2. increased signaling **activity** of the same secreted protein in a
   *receiver* cell type.

The pipeline has three steps: (1) per-cell-type secreted-protein differential
expression (Wilcoxon rank-sum), (2) per-cell-type differential signaling
**activity** via permutation-ridge inference, and (3) linking up-expressed
senders to up-active receivers, scoring each sender→protein→receiver edge by
``sender_exp_logFC × receiver_act_diff`` with a Fisher-combined p-value.

FlashReg note
-------------
The only compute-heavy numerical kernel is step 2's permutation ridge, which is
delegated to :func:`secactpy.secact_activity_inference` and therefore runs on
whatever accelerated backend that function is given (numpy / cupy / cuda_native
/ flashregpy). This module adds **no new FlashReg kernel** — it is orchestration
over the existing accelerated activity inference. Step 1's Wilcoxon test is a
rank-sum (not a ridge), vectorized here with SciPy; step 3 is lightweight
DataFrame work.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .inference import secact_activity_inference
from .signature import load_signature

__all__ = ["secact_ccc_scrnaseq"]


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (matches R p.adjust method='BH'); NaNs pass through."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def _extract_counts(adata, cell_type_col, condition_col):
    """AnnData -> (counts genes×cells, gene_names, obs DataFrame).

    Prefers raw counts (``adata.raw.X`` then ``adata.X``); uppercases and
    version-strips gene symbols and drops duplicate symbols (keeping the
    highest-mean copy), mirroring ``secact_activity_inference_scrnaseq``.
    """
    if isinstance(adata, str):
        import anndata
        adata = anndata.read_h5ad(adata)

    if getattr(adata, "raw", None) is not None:
        X = adata.raw.X
        gene_names = list(adata.raw.var_names)
    else:
        X = adata.X
        gene_names = list(adata.var_names)

    # AnnData is cells × genes -> transpose to genes × cells
    counts = X.T
    counts = counts.tocsr() if sparse.issparse(counts) else np.asarray(counts, dtype=np.float64)

    obs = adata.obs.copy()
    for col in (cell_type_col, condition_col):
        if col is not None and col not in obs.columns:
            raise ValueError(f"Column '{col}' not in adata.obs (have {list(obs.columns)}).")

    gene_names = [g.upper().split(".")[0] for g in gene_names]
    if len(gene_names) != len(set(gene_names)):
        gene_means = (np.asarray(counts.mean(axis=1)).ravel()
                      if sparse.issparse(counts) else counts.mean(axis=1))
        best: dict[str, int] = {}
        for idx, g in enumerate(gene_names):
            if g not in best or gene_means[idx] > gene_means[best[g]]:
                best[g] = idx
        keep = sorted(best.values())
        counts = counts[keep, :]
        gene_names = [gene_names[i] for i in keep]

    return counts, np.asarray(gene_names, dtype=object), obs


def _col_slice(counts, idx):
    """Dense genes×len(idx) block for a set of cell (column) indices."""
    sub = counts[:, idx]
    return sub.toarray() if sparse.issparse(sub) else np.asarray(sub)


def _normalize_log(block, scale_factor):
    """Per-cell normalize to scale_factor then log2(x+1) (== R normalize_log_sparse)."""
    col_sums = block.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    return np.log2(block / col_sums * scale_factor + 1.0)


def secact_ccc_scrnaseq(
    adata: Any,
    cell_type_col: str,
    condition_col: Optional[str] = None,
    condition_case: Optional[str] = None,
    condition_control: Optional[str] = None,
    *,
    scale_factor: float = 1e5,
    act_diff_cutoff: float = 2.0,
    exp_logfc_cutoff: float = 0.2,
    exp_mean_all_cutoff: float = 2.0,
    exp_fraction_case_cutoff: float = 0.1,
    padj_cutoff: float = 0.01,
    min_cells: int = 30,
    sig_matrix: str = "secact",
    is_group_sig: Optional[bool] = None,
    is_group_cor: float = 0.9,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    backend: str = "auto",
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Infer secreted-protein cell-cell communication from scRNA-seq.

    Port of R ``SecAct.CCC.scRNAseq``. If ``condition_col`` is given, CCC is
    condition-specific (``condition_case`` vs ``condition_control``); otherwise
    each cell type is compared against all other cells.

    Parameters
    ----------
    adata : AnnData or str
        AnnData (cells × genes; raw counts in ``.raw.X`` or ``.X``) or path to
        an ``.h5ad``.
    cell_type_col, condition_col : str
        ``adata.obs`` columns for cell type and (optional) condition.
    condition_case, condition_control : str
        Condition labels, required when ``condition_col`` is set.
    scale_factor, act_diff_cutoff, exp_logfc_cutoff, exp_mean_all_cutoff, \
exp_fraction_case_cutoff, padj_cutoff, min_cells
        Same cutoffs/defaults as the R function.
    sig_matrix, is_group_sig, is_group_cor, lambda_, n_rand, backend, seed
        Passed to :func:`secactpy.secact_activity_inference` for step 2 (the
        accelerated permutation-ridge; ``backend`` selects numpy/cupy/
        cuda_native).

    Returns
    -------
    dict with keys
        ``secreted_protein_expression`` : {cell_type -> DataFrame} of per-SP
        expression stats; ``secreted_protein_activity`` : the activity-inference
        result (``zscore``/``pvalue`` DataFrames, SP × cell type);
        ``secreted_protein_ccc`` : DataFrame of significant sender→protein→
        receiver edges with strengths and combined p-values.
    """
    if condition_col is not None and (condition_case is None or condition_control is None):
        raise ValueError("condition_case and condition_control are required when condition_col is set.")

    counts, gene_names, obs = _extract_counts(adata, cell_type_col, condition_col)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    ct_vec = obs[cell_type_col].astype(str).values
    cond_vec = obs[condition_col].astype(str).values if condition_col else None

    # Secreted-protein names = signature columns
    sig = load_signature(sig_matrix) if isinstance(sig_matrix, str) else sig_matrix
    sps = [s for s in (sig.columns if hasattr(sig, "columns") else sig)]
    sp_upper = [str(s).upper() for s in sps]

    # cell types present (in both conditions when applicable)
    if condition_col is None:
        cell_types = sorted(set(ct_vec), key=str.lower)
    else:
        case_cts = set(ct_vec[cond_vec == str(condition_case)])
        ctrl_cts = set(ct_vec[cond_vec == str(condition_control)])
        cell_types = sorted(case_cts & ctrl_cts, key=str.lower)

    def _case_control_idx(ct):
        if condition_col is None:
            case = np.where(ct_vec == ct)[0]
            control = np.where(ct_vec != ct)[0]
        else:
            case = np.where((cond_vec == str(condition_case)) & (ct_vec == ct))[0]
            control = np.where((cond_vec == str(condition_control)) & (ct_vec == ct))[0]
        return case, control

    # SP genes actually measured
    sp_gene_rows = [(g, gene_index[g]) for g in sp_upper if g in gene_index]
    sp_gene_names = [g for g, _ in sp_gene_rows]
    sp_gene_idx = np.asarray([i for _, i in sp_gene_rows], dtype=int)

    # ---- Step 1: per-cell-type secreted-protein differential expression ----
    if verbose:
        print("Step 1: assessing changes in secreted protein expression.")
    expression: dict[str, pd.DataFrame] = {}
    for ct in cell_types:
        case, control = _case_control_idx(ct)
        if case.size < min_cells:
            continue
        case_mat = _normalize_log(_col_slice(counts, case)[sp_gene_idx], scale_factor)
        ctrl_mat = _normalize_log(_col_slice(counts, control)[sp_gene_idx], scale_factor)

        mean_case = case_mat.mean(axis=1)
        mean_control = ctrl_mat.mean(axis=1)
        n_c, n_k = case_mat.shape[1], ctrl_mat.shape[1]
        mean_all = (mean_case * n_c + mean_control * n_k) / (n_c + n_k)
        with np.errstate(invalid="ignore"):
            pvals = stats.mannwhitneyu(case_mat, ctrl_mat, axis=1,
                                       alternative="two-sided").pvalue
        df = pd.DataFrame({
            "exp_logFC": mean_case - mean_control,
            "exp_mean_all": mean_all,
            "exp_mean_case": mean_case,
            "exp_mean_control": mean_control,
            "exp_fraction_case": (case_mat > 0).mean(axis=1),
            "exp_fraction_control": (ctrl_mat > 0).mean(axis=1),
            "exp_pv": pvals,
        }, index=sp_gene_names)
        df = df[df["exp_mean_all"] > 0]
        df["exp_pv.adj"] = _benjamini_hochberg(df["exp_pv"].values)
        expression[ct] = df.sort_values("exp_pv.adj")

    if len(expression) < 2:
        raise ValueError("Fewer than two cell types have >= min_cells cells; cannot infer CCC.")

    # ---- Step 2: per-cell-type differential activity (permutation ridge) ----
    if verbose:
        print("Step 2: calculating changes in secreted protein activity (permutation ridge).")
    kept_cts = list(expression.keys())
    bulk_diff = pd.DataFrame(index=gene_names, dtype=float)
    for ct in kept_cts:
        case, control = _case_control_idx(ct)

        def _pseudobulk_log2tpm(idx):
            block = _col_slice(counts, idx)
            summed = block.sum(axis=1)
            total = summed.sum()
            tpm = summed / (total if total else 1.0) * 1e6
            return np.log2(tpm + 1.0)

        bulk_diff[ct] = _pseudobulk_log2tpm(case) - _pseudobulk_log2tpm(control)

    activity = secact_activity_inference(
        bulk_diff, is_differential=True, sig_matrix=sig_matrix,
        is_group_sig=is_group_sig, is_group_cor=is_group_cor,
        lambda_=lambda_, n_rand=n_rand, backend=backend, seed=seed,
        verbose=verbose,
    )
    zscore, pvalue = activity["zscore"], activity["pvalue"]

    # ---- Step 3: link up-expressed senders to up-active receivers ----
    if verbose:
        print("Step 3: linking source and receiver cell types.")

    senders = []   # rows: (cell_type, SP, exp_logFC, exp_pv, exp_pv.adj)
    for ct in kept_cts:
        df = expression[ct]
        up = df[(df["exp_logFC"] > exp_logfc_cutoff)
                & (df["exp_mean_all"] > exp_mean_all_cutoff)
                & (df["exp_fraction_case"] > exp_fraction_case_cutoff)
                & (df["exp_pv.adj"] < padj_cutoff)]
        for sp, row in up.iterrows():
            senders.append((ct, sp, row["exp_logFC"], row["exp_pv"], row["exp_pv.adj"]))

    # receiver candidates: SP × cell type with up activity
    receivers: dict[str, list] = {}
    for ct in [c for c in kept_cts if c in zscore.columns]:
        act_pv_adj = pd.Series(_benjamini_hochberg(pvalue[ct].values), index=pvalue.index)
        for sp in zscore.index:
            zd = float(zscore.loc[sp, ct])
            if zd > act_diff_cutoff and float(act_pv_adj[sp]) < padj_cutoff:
                receivers.setdefault(str(sp).upper(), []).append(
                    (ct, zd, float(pvalue.loc[sp, ct]), float(act_pv_adj[sp])))

    rows = []
    for sender_ct, sp, exp_logfc, exp_pv, exp_pv_adj in senders:
        for recv_ct, act_diff, act_pv, act_pv_adj in receivers.get(str(sp).upper(), []):
            if recv_ct == sender_ct:
                continue
            rows.append({
                "sender": sender_ct, "secretedProtein": sp, "receiver": recv_ct,
                "sender_exp_logFC": exp_logfc, "sender_exp_pv": exp_pv,
                "sender_exp_pv.adj": exp_pv_adj, "receiver_act_diff": act_diff,
                "receiver_act_pv": act_pv, "receiver_act_pv.adj": act_pv_adj,
            })

    ccc = pd.DataFrame(rows)
    if not ccc.empty:
        ccc["overall_strength"] = ccc["sender_exp_logFC"] * ccc["receiver_act_diff"]
        tiny = np.finfo(float).tiny
        combined = [
            stats.combine_pvalues([max(a, tiny), max(b, tiny)], method="fisher").pvalue
            for a, b in zip(ccc["sender_exp_pv"], ccc["receiver_act_pv"])
        ]
        ccc["overall_pv"] = combined
        ccc["overall_pv.adj"] = _benjamini_hochberg(ccc["overall_pv"].values)
        ccc = ccc[ccc["overall_pv.adj"] < padj_cutoff].sort_values("overall_pv.adj").reset_index(drop=True)

    return {
        "secreted_protein_expression": expression,
        "secreted_protein_activity": activity,
        "secreted_protein_ccc": ccc,
    }
