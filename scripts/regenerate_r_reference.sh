#!/bin/bash
#SBATCH --job-name=regen_ref
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --output=regen_ref_%j.out

# Regenerate R reference output for cross-path tests.
# This ensures the reference matches this platform's C rand() implementation.

module load R

cd /vf/users/parks34/projects/1ridgesig/SecActpy

# Install SecAct from source + RidgeFast accelerator if not already installed.
# RidgeFast is optional but matches the production Docker image, ensuring this
# script reproduces the same numerics as `psychemistz/secactpy:with-r`.
# RidgeFast itself depends on system GSL (apt: libgsl-dev / brew: gsl).
Rscript -e '
# Force install from local source to ensure latest version
cat("Installing SecAct from local source...\n")
install.packages(
    "/vf/users/parks34/projects/1ridgesig/SecAct",
    repos = NULL, type = "source", INSTALL_opts = "--no-multiarch"
)
library(SecAct)
cat("SecAct loaded. Functions:", paste(ls("package:SecAct"), collapse=", "), "\n")

# Optional CPU accelerator (skip silently if unavailable so the script can
# still run on a minimal R install — SecAct falls back to its pure-R path).
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org/")
}
tryCatch({
    if (!requireNamespace("RidgeFast", quietly = TRUE)) {
        cat("Installing RidgeFast (CPU accelerator)...\n")
        remotes::install_github("data2intelligence/RidgeFast",
                                dependencies = NA, force = FALSE)
    }
    library(RidgeFast)
    cat("RidgeFast", as.character(packageVersion("RidgeFast")), "loaded\n")
}, error = function(e) {
    message("RidgeFast install/load failed: ", conditionMessage(e))
    message("Falling back to SecAct pure-R path (slower but correct).")
})

dataPath <- file.path(system.file(package = "SecAct"), "extdata")
expr <- read.table(paste0(dataPath, "/Ly86-Fc_vs_Vehicle_logFC.txt"))

cat("Running SecAct.inference.gsl...\n")
cat(sprintf("Input: %d genes x %d samples\n", nrow(expr), ncol(expr)))
cat(sprintf("Platform: %s\n", R.version$platform))
cat(sprintf("R version: %s\n", R.version.string))

res <- SecAct.inference.gsl(expr)

cat(sprintf("Output: %d features x %d samples\n", nrow(res$beta), ncol(res$beta)))

outdir <- "dataset/output/ridge/bulk"
write.table(res$beta,   file.path(outdir, "beta.txt"),   quote=FALSE)
write.table(res$se,     file.path(outdir, "se.txt"),     quote=FALSE)
write.table(res$zscore, file.path(outdir, "zscore.txt"), quote=FALSE)
write.table(res$pvalue, file.path(outdir, "pvalue.txt"), quote=FALSE)

cat("Reference files written to:", outdir, "\n")
cat("SE range:", range(res$se), "\n")
cat("First 3 SE values:", head(res$se, 3), "\n")
'

# Now verify Python matches
echo ""
echo "=== Verifying Python matches fresh R reference ==="
python -c "
import numpy as np, pandas as pd
from secactpy import load_signature
from secactpy.ridge import ridge

sig_df = load_signature('secact')
Y_df = pd.read_csv('dataset/input/Ly86-Fc_vs_Vehicle_logFC.txt', sep=r'\s+', index_col=0)
common_genes = Y_df.index.intersection(sig_df.index)
X_aligned = sig_df.loc[common_genes].astype(np.float64)
Y_aligned = Y_df.loc[common_genes].astype(np.float64)
X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std(ddof=1)
Y_scaled = (Y_aligned - Y_aligned.mean()) / Y_aligned.std(ddof=1)
X = X_scaled.fillna(0).values
Y = Y_scaled.fillna(0).values

result = ridge(X, Y, lambda_=5e5, n_rand=1000, seed=0, backend='numpy', use_cache=False)

r_dir = 'dataset/output/ridge/bulk'
features = X_scaled.columns.tolist()
samples = Y_scaled.columns.tolist()

for name in ['beta', 'se', 'zscore', 'pvalue']:
    py_df = pd.DataFrame(result[name], index=features, columns=samples)
    r_df = pd.read_csv(f'{r_dir}/{name}.txt', sep=r'\s+', index_col=0)
    py_aligned = py_df.loc[r_df.index, r_df.columns]
    diff = np.abs(py_aligned.values - r_df.values)
    max_diff = np.nanmax(diff)
    status = 'PASS' if max_diff < 1e-8 else 'FAIL'
    print(f'  {name:8s}: max diff = {max_diff:.2e}  {status}')
"
