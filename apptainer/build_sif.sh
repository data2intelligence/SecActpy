#!/bin/bash
# =============================================================================
# Build SecActPy SIF images via SLURM
#
# Usage:
#   sbatch build_sif.sh cpu        # Python-only CPU image
#   sbatch build_sif.sh gpu        # Python + CuPy GPU image
#   sbatch build_sif.sh cpu-r      # Python + R CPU image
#   sbatch build_sif.sh gpu-r      # Python + CuPy + R GPU image
#
# The SIF file is written to OUTPUT_DIR (default: directory of this script).
# Override with: sbatch --export=OUTPUT_DIR=/path/to/dir build_sif.sh cpu
# =============================================================================

#SBATCH --job-name=secactpy-build
#SBATCH --output=secactpy-build-%j.out
#SBATCH --error=secactpy-build-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=norm

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse variant argument
# ---------------------------------------------------------------------------
VARIANT="${1:-}"
if [ -z "$VARIANT" ]; then
    echo "ERROR: No variant specified."
    echo "Usage: sbatch build_sif.sh [cpu|gpu|cpu-r|gpu-r]"
    exit 1
fi

case "$VARIANT" in
    cpu|gpu|cpu-r|gpu-r) ;;
    *)
        echo "ERROR: Unknown variant '$VARIANT'."
        echo "Valid variants: cpu, gpu, cpu-r, gpu-r"
        exit 1
        ;;
esac

# Increase walltime for R builds (they take much longer)
case "$VARIANT" in
    *-r)
        # Note: SLURM walltime is set at submission. If you need more time for
        # R builds, submit with: sbatch --time=08:00:00 build_sif.sh cpu-r
        echo "INFO: R builds may take 2-4 hours. If this times out, resubmit with:"
        echo "  sbatch --time=08:00:00 build_sif.sh $VARIANT"
        ;;
esac

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/secactpy-${VARIANT}.def"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}}"
SIF_FILE="${OUTPUT_DIR}/secactpy-${VARIANT}.sif"

if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Apptainer cache/tmp on local scratch to avoid home quota issues
# ---------------------------------------------------------------------------
if [ -n "${TMPDIR:-}" ]; then
    SCRATCH="$TMPDIR"
elif [ -d "/lscratch/${SLURM_JOB_ID}" ]; then
    SCRATCH="/lscratch/${SLURM_JOB_ID}"
else
    SCRATCH=$(mktemp -d)
fi

export APPTAINER_TMPDIR="${SCRATCH}/apptainer_tmp"
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "============================================================"
echo "SecActPy SIF Build"
echo "============================================================"
echo "Variant:    $VARIANT"
echo "Def file:   $DEF_FILE"
echo "Output:     $SIF_FILE"
echo "Scratch:    $SCRATCH"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory:     ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Start time: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
apptainer build --fakeroot "$SIF_FILE" "$DEF_FILE"

echo ""
echo "============================================================"
echo "Build completed: $(date)"
echo "SIF file: $SIF_FILE ($(du -h "$SIF_FILE" | cut -f1))"
echo "============================================================"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "Verifying Python..."
apptainer exec "$SIF_FILE" python3 -c "import secactpy; print(f'SecActPy {secactpy.__version__} OK, GPU: {secactpy.CUPY_AVAILABLE}')"

case "$VARIANT" in
    *-r)
        echo "Verifying R..."
        apptainer exec "$SIF_FILE" R -e "cat('R version:', R.version.string, '\n'); library(SecAct); cat('SecAct OK\n'); library(RidgeR); cat('RidgeR OK\n')"
        ;;
esac

echo ""
echo "All verifications passed."
