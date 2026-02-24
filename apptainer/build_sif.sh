#!/bin/bash
# =============================================================================
# Build SecActPy SIF images from Docker Hub
#
# Pulls pre-built images from Docker Hub and converts them to SIF format.
# No root or fakeroot required. Can run on a login node or via SLURM.
#
# Usage (login node):
#   ./build_sif.sh cpu           # Python-only CPU image
#   ./build_sif.sh gpu           # Python + CuPy GPU image
#   ./build_sif.sh cpu-r         # Python + R CPU image
#   ./build_sif.sh gpu-r         # Python + CuPy + R GPU image
#   ./build_sif.sh all           # All 4 variants
#
# Usage (SLURM):
#   sbatch build_sif.sh cpu
#   sbatch build_sif.sh all
#
# The SIF file is written to OUTPUT_DIR (default: directory of this script).
# Override with: OUTPUT_DIR=/path/to/dir ./build_sif.sh cpu
# =============================================================================

#SBATCH --job-name=secactpy-build
#SBATCH --output=secactpy-build-%j.out
#SBATCH --error=secactpy-build-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=norm

set -euo pipefail

# Load apptainer module if available (required on HPC nodes)
if command -v module &>/dev/null; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi

# Docker Hub repository
DOCKER_REPO="psychemistz/secactpy"

# Map variant names to Docker Hub tags
declare -A DOCKER_TAGS=(
    [cpu]="cpu"
    [gpu]="gpu"
    [cpu-r]="cpu-with-r"
    [gpu-r]="gpu-with-r"
)

# ---------------------------------------------------------------------------
# Parse variant argument
# ---------------------------------------------------------------------------
VARIANT="${1:-}"
if [ -z "$VARIANT" ]; then
    echo "ERROR: No variant specified."
    echo "Usage: ./build_sif.sh [cpu|gpu|cpu-r|gpu-r|all]"
    exit 1
fi

ALL_VARIANTS=(cpu gpu cpu-r gpu-r)

if [ "$VARIANT" = "all" ]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
else
    case "$VARIANT" in
        cpu|gpu|cpu-r|gpu-r) VARIANTS=("$VARIANT") ;;
        *)
            echo "ERROR: Unknown variant '$VARIANT'."
            echo "Valid variants: cpu, gpu, cpu-r, gpu-r, all"
            exit 1
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -d "${SLURM_SUBMIT_DIR}/apptainer" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/apptainer"
    else
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}}"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Apptainer cache/tmp — use local scratch to avoid home quota issues
# ---------------------------------------------------------------------------
if [ -n "${TMPDIR:-}" ]; then
    SCRATCH="$TMPDIR"
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/lscratch/${SLURM_JOB_ID}" ]; then
    SCRATCH="/lscratch/${SLURM_JOB_ID}"
else
    SCRATCH=$(mktemp -d)
fi

export APPTAINER_TMPDIR="${SCRATCH}/apptainer_tmp"
export APPTAINER_CACHEDIR="${SCRATCH}/apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ---------------------------------------------------------------------------
# Build each variant
# ---------------------------------------------------------------------------
for V in "${VARIANTS[@]}"; do
    TAG="${DOCKER_TAGS[$V]}"
    SIF_FILE="${OUTPUT_DIR}/secactpy-${V}.sif"
    DOCKER_URI="docker://${DOCKER_REPO}:${TAG}"

    echo "============================================================"
    echo "SecActPy SIF Build"
    echo "============================================================"
    echo "Variant:    $V"
    echo "Source:     $DOCKER_URI"
    echo "Output:     $SIF_FILE"
    echo "Scratch:    $SCRATCH"
    echo "Start time: $(date)"
    echo "============================================================"

    apptainer build "$SIF_FILE" "$DOCKER_URI"

    echo ""
    echo "============================================================"
    echo "Build completed: $(date)"
    echo "SIF file: $SIF_FILE ($(du -h "$SIF_FILE" | cut -f1))"
    echo "============================================================"

    # Verify
    echo ""
    echo "Verifying Python..."
    apptainer exec "$SIF_FILE" python3 -c "import secactpy; print(f'SecActPy {secactpy.__version__} OK, GPU: {secactpy.CUPY_AVAILABLE}')"

    case "$V" in
        *-r)
            echo "Verifying R..."
            apptainer exec "$SIF_FILE" R -e "cat('R version:', R.version.string, '\n'); library(SecAct); cat('SecAct OK\n'); library(RidgeR); cat('RidgeR OK\n')"
            ;;
    esac

    echo ""
    echo "All verifications passed for $V."
    echo ""
done
