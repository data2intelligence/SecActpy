"""Streaming path vs in-memory path: the same numbers, not just the same intent.

`streaming.py` states "Results are numerically identical to the non-streaming path."
`test_cross_path.py` pins six code paths against each other and against R, but the
streaming path is not among them -- it was added after that file was written. This
closes that gap, so the docstring's claim is defended by a check rather than asserted.

What could break and would be caught here: any change to
`_StreamingStatsAccumulator.accumulate` / `.finalize`, or to the pass-2 cross term,
that made the reference anything other than the row means over EVERY cell in the file.
A per-batch reference is the specific failure mode -- it would leave a cell's activity
depending on which chunk it landed in, and nothing else in the suite would notice.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

ad = pytest.importorskip("anndata")

from secactpy.inference import secact_activity_inference_scrnaseq  # noqa: E402

N_GENES, N_CELLS, N_SIG = 400, 300, 8
TOL = 1e-10


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Counts with a deliberate split: the halves have different mean profiles.

    Without that imbalance a per-batch reference would agree with a global one by
    accident, and the test would pass while measuring nothing.
    """
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(N_GENES)]
    X = pd.DataFrame(np.abs(rng.normal(size=(N_GENES, N_SIG))), index=genes,
                     columns=[f"SIG{k}" for k in range(N_SIG)])
    counts = rng.poisson(2.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    counts[: N_CELLS // 2] += rng.poisson(6.0, size=(1, N_GENES)).astype(np.float32)
    A = ad.AnnData(
        X=sps.csr_matrix(counts),
        obs=pd.DataFrame({"cell_type": rng.choice(["T", "B", "M"], N_CELLS)},
                         index=[f"C{j}" for j in range(N_CELLS)]),
        var=pd.DataFrame(index=genes),
    )
    path = tmp_path_factory.mktemp("stream") / "data.h5ad"
    A.write_h5ad(path)
    return A, str(path), X


def _kwargs(sig):
    return dict(cell_type_col="cell_type", sig_matrix=sig, is_single_cell_level=True,
                n_rand=5, lambda_=5e4, seed=0, verbose=False, is_group_sig=False)


# Small batch sizes are the ones with detection power. Verified by injecting a
# per-batch reference into pass 2: batch_size 64 and 128 failed, 4096 did NOT --
# 4096 > N_CELLS means a single batch, where a per-batch reference IS the global
# one. A test parametrised only on large batches would pass against the bug.
@pytest.mark.parametrize("batch_size", [64, 128, 4096])
def test_streaming_matches_in_memory(dataset, batch_size):
    A, path, X = dataset
    mem = secact_activity_inference_scrnaseq(A, batch_size=batch_size, **_kwargs(X))
    strm = secact_activity_inference_scrnaseq(path, streaming=True,
                                              batch_size=batch_size, **_kwargs(X))
    for key in ("beta", "se", "zscore", "pvalue"):
        a = mem[key]
        b = strm[key].reindex(index=a.index, columns=a.columns)
        np.testing.assert_allclose(
            b.to_numpy(float), a.to_numpy(float), rtol=TOL, atol=TOL,
            err_msg=f"streaming vs in-memory mismatch in {key} at batch_size={batch_size}")


def test_streaming_reference_is_global(dataset):
    """The accumulated reference must equal the one a single batch would compute.

    This is the property everything else rests on, so it is checked directly rather
    than only through the end-to-end result.
    """
    from secactpy.batch import _compute_population_stats
    from secactpy.streaming import _StreamingStatsAccumulator

    rng = np.random.default_rng(1)
    Y = sps.csc_matrix(np.abs(rng.normal(size=(N_GENES, N_CELLS)))
                       * (rng.random((N_GENES, N_CELLS)) < 0.3))
    ref = _compute_population_stats(Y, ddof=1, row_center=True)
    for chunk in (7, 64, N_CELLS):
        acc = _StreamingStatsAccumulator(n_genes=N_GENES)
        for j in range(0, N_CELLS, chunk):
            acc.accumulate(Y[:, j:j + chunk])
        row_means, _, _, mu, _, _ = acc.finalize()
        np.testing.assert_allclose(row_means, ref.row_means, rtol=0, atol=1e-12,
                                   err_msg=f"reference differs at chunk={chunk}")
        np.testing.assert_allclose(mu, ref.mu, rtol=0, atol=1e-12,
                                   err_msg=f"column means differ at chunk={chunk}")
