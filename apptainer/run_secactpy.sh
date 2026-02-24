#!/bin/bash
# =============================================================================
# Run SecActPy via Apptainer on HPC
#
# Usage:
#   ./run_secactpy.sh shell [cpu|gpu|cpu-r|gpu-r]    Interactive shell
#   ./run_secactpy.sh exec  [cpu|gpu|cpu-r|gpu-r] CMD Run a command
#   ./run_secactpy.sh jupyter [cpu|gpu|cpu-r|gpu-r]  Start Jupyter Lab
#   ./run_secactpy.sh sbatch [cpu|gpu|cpu-r|gpu-r] script.py  Submit SLURM job
#
# Environment variables:
#   SIF_DIR   Directory containing SIF files (default: script directory)
#   BIND      Extra bind mounts (default: none, added to standard mounts)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_DIR="${SIF_DIR:-${SCRIPT_DIR}}"

usage() {
    echo "Usage: $0 <mode> <variant> [args...]"
    echo ""
    echo "Modes:"
    echo "  shell   - Interactive shell"
    echo "  exec    - Run a command (pass command after variant)"
    echo "  jupyter - Start Jupyter Lab (prints connection instructions)"
    echo "  sbatch  - Submit a Python script as a SLURM batch job"
    echo ""
    echo "Variants: cpu, gpu, cpu-r, gpu-r"
    echo ""
    echo "Examples:"
    echo "  $0 shell cpu-r"
    echo "  $0 exec cpu python3 my_analysis.py"
    echo "  $0 exec gpu-r R -e \"library(SecAct)\""
    echo "  $0 jupyter gpu"
    echo "  $0 sbatch cpu my_script.py"
    exit 1
}

MODE="${1:-}"
VARIANT="${2:-}"

if [ -z "$MODE" ] || [ -z "$VARIANT" ]; then
    usage
fi

# Validate variant
case "$VARIANT" in
    cpu|gpu|cpu-r|gpu-r) ;;
    *) echo "ERROR: Unknown variant '$VARIANT'"; usage ;;
esac

SIF="${SIF_DIR}/secactpy-${VARIANT}.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: SIF file not found: $SIF"
    echo "Build it first with: sbatch build_sif.sh $VARIANT"
    exit 1
fi

# GPU flag for apptainer
GPU_FLAG=""
case "$VARIANT" in
    gpu|gpu-r) GPU_FLAG="--nv" ;;
esac

# Standard bind mounts: current directory as /workspace
BIND_ARGS="--bind $(pwd):/workspace"
if [ -n "${BIND:-}" ]; then
    BIND_ARGS="$BIND_ARGS --bind $BIND"
fi

shift 2  # Remove mode and variant from args

case "$MODE" in
    shell)
        echo "Starting interactive shell (variant: $VARIANT)..."
        apptainer shell $GPU_FLAG $BIND_ARGS "$SIF"
        ;;

    exec)
        if [ $# -eq 0 ]; then
            echo "ERROR: No command specified for exec mode"
            usage
        fi
        apptainer exec $GPU_FLAG $BIND_ARGS "$SIF" "$@"
        ;;

    jupyter)
        PORT="${1:-8888}"
        echo "Starting Jupyter Lab on port $PORT..."
        echo ""
        echo "Connect via SSH tunnel:"
        echo "  ssh -L ${PORT}:localhost:${PORT} $(hostname)"
        echo ""
        apptainer exec $GPU_FLAG $BIND_ARGS "$SIF" \
            jupyter lab --ip=0.0.0.0 --port="$PORT" --no-browser --allow-root
        ;;

    sbatch)
        if [ $# -eq 0 ]; then
            echo "ERROR: No script specified for sbatch mode"
            usage
        fi
        SCRIPT="$1"
        shift
        # Submit a SLURM job that runs the script inside the container
        sbatch --wrap="apptainer exec $GPU_FLAG $BIND_ARGS $SIF python3 /workspace/$(basename "$SCRIPT") $*" \
            --job-name="secactpy-$(basename "$SCRIPT" .py)" \
            --output="secactpy-$(basename "$SCRIPT" .py)-%j.out" \
            --cpus-per-task=4 \
            --mem=16G \
            --time=04:00:00
        ;;

    *)
        echo "ERROR: Unknown mode '$MODE'"
        usage
        ;;
esac
