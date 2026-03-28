#!/bin/bash
#SBATCH --job-name=test_xpath
#SBATCH --partition=norm
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=test_xpath_%j.out

cd /vf/users/parks34/projects/1ridgesig/SecActpy

# Clear any stale perm table caches
rm -f ~/.cache/secactpy/perm_tables/inv_perm_n7720_nperm1000_seed0.npy

python -m pytest tests/test_cross_path.py tests/test_ridge.py tests/test_rng_gsl.py tests/test_col_normalization.py tests/test_sparse_mode.py -v 2>&1
