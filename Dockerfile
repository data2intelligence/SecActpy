# =============================================================================
# SecActPy + SecAct/RidgeR Unified Docker Image
#
# Single Dockerfile for both CPU and GPU versions
#
# Build CPU version (default, Python only):
#   docker build -t secactpy:latest .
#
# Build GPU version:
#   docker build -t secactpy:gpu --build-arg USE_GPU=true .
#
# Build with R SecAct/RidgeR package (slower):
#   docker build -t secactpy:with-r --build-arg INSTALL_R=true .
#   docker build -t secactpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
#
# Run CPU:
#   docker run -it --rm -v $(pwd):/workspace secactpy:latest
#
# Run GPU:
#   docker run -it --rm --gpus all -v $(pwd):/workspace secactpy:gpu
#
# Run Jupyter:
#   docker run -it --rm -p 8888:8888 -v $(pwd):/workspace secactpy:latest \
#       jupyter lab --ip=0.0.0.0 --no-browser --allow-root
# =============================================================================

# Build arguments
ARG USE_GPU=false
ARG INSTALL_R=false
ARG GITHUB_PAT=""

# =============================================================================
# Base Image Selection
# =============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base-true
FROM ubuntu:22.04 AS base-false

# Select base image based on USE_GPU argument
FROM base-${USE_GPU} AS base

# Re-declare ARGs after FROM (required by Docker)
ARG USE_GPU=false
ARG INSTALL_R=false

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Build tools
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    # Libraries for R packages
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgsl-dev \
    libhdf5-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libxt-dev \
    libmagick++-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    # For qqconf (dependency of metap)
    libfftw3-dev \
    # For RcppParallel (Intel TBB)
    libtbb-dev \
    # BLAS/LAPACK (reference + OpenBLAS for multi-threaded linear algebra)
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    # Utilities
    git \
    wget \
    curl \
    vim \
    locales \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set locale (needed for some R packages)
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# R: Install R and Packages (optional)
# =============================================================================

ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing R from CRAN repository..." && \
        echo "========================================" && \
        wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
            | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
        echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
            | tee /etc/apt/sources.list.d/cran.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            r-base \
            r-base-dev && \
        rm -rf /var/lib/apt/lists/* && \
        R -e "cat('R version:', R.version.string, '\n')"; \
    else \
        echo "Skipping R installation"; \
    fi

# Switch R's BLAS/LAPACK to OpenBLAS for multi-threaded linear algebra
# Reference BLAS is single-threaded; OpenBLAS uses all available cores
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Switching R to OpenBLAS..." && \
        echo "========================================" && \
        ARCH=$(dpkg --print-architecture) && \
        MULTIARCH=$(dpkg-architecture -qDEB_HOST_MULTIARCH) && \
        OPENBLAS_BLAS="/usr/lib/${MULTIARCH}/openblas-pthread/libblas.so.3" && \
        OPENBLAS_LAPACK="/usr/lib/${MULTIARCH}/openblas-pthread/liblapack.so.3" && \
        if [ -f "$OPENBLAS_BLAS" ]; then \
            update-alternatives --set "libblas.so.3-${ARCH}" "$OPENBLAS_BLAS" && \
            update-alternatives --set "liblapack.so.3-${ARCH}" "$OPENBLAS_LAPACK" && \
            echo "OpenBLAS configured for ${ARCH}"; \
        else \
            echo "WARNING: OpenBLAS not found at $OPENBLAS_BLAS, using reference BLAS"; \
        fi && \
        R -e "si <- sessionInfo(); cat('BLAS:', si\$BLAS, '\nLAPACK:', si\$LAPACK, '\n')"; \
    fi

# Tell arrow to download pre-built C++ binaries instead of compiling from source
ENV NOT_CRAN=true
ENV LIBARROW_BINARY=true

# Configure Posit Package Manager (RSPM) for pre-compiled R binaries.
# RSPM provides pre-compiled binary packages for x86_64 (amd64) Ubuntu only.
# On arm64, packages must be compiled from source via CRAN, so we leave RSPM
# unset and the R install commands fall back to https://cloud.r-project.org/.
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        ARCH=$(dpkg --print-architecture) && \
        if [ "$ARCH" = "amd64" ]; then \
            echo 'RSPM=https://packagemanager.posit.co/cran/__linux__/jammy/latest' \
                >> "$(R RHOME)/etc/Renviron.site" && \
            echo "Configured RSPM for pre-compiled R binaries (amd64)"; \
        else \
            echo "Using CRAN source packages ($ARCH - no RSPM binaries available)"; \
        fi; \
    fi

# Install Bioconductor packages FIRST (required by CRAN packages like NMF,
# and by SecAct, SpaCET, and RidgeR)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing BiocManager and Bioconductor packages..." && \
        echo "========================================" && \
        R -e "options(timeout = 600, repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'))); \
              install.packages(c('remotes', 'BiocManager'), Ncpus = parallel::detectCores())" && \
        R -e "options(timeout = 600); BiocManager::install(ask = FALSE, update = FALSE)" && \
        R -e "options(timeout = 600); \
              bioc_pkgs <- c( \
                  'Biobase', 'S4Vectors', 'IRanges', \
                  'SummarizedExperiment', 'SingleCellExperiment', \
                  'rhdf5', 'ComplexHeatmap', 'limma', \
                  'UCell', 'BiRewire', \
                  'sva', 'multtest' \
              ); \
              BiocManager::install(bioc_pkgs, ask = FALSE, update = FALSE, \
                  Ncpus = parallel::detectCores()); \
              missing <- bioc_pkgs[!sapply(bioc_pkgs, requireNamespace, quietly = TRUE)]; \
              if (length(missing) > 0) { \
                  cat('WARNING: Bioconductor packages failed on first attempt:', \
                      paste(missing, collapse=', '), '\n'); \
                  cat('Retrying individually...\n'); \
                  for (pkg in missing) { \
                      tryCatch( \
                          BiocManager::install(pkg, ask = FALSE, update = FALSE, \
                              Ncpus = parallel::detectCores()), \
                          error = function(e) cat('  ERROR installing', pkg, ':', \
                              conditionMessage(e), '\n') \
                      ) \
                  } \
              }; \
              still_missing <- bioc_pkgs[!sapply(bioc_pkgs, requireNamespace, quietly = TRUE)]; \
              if (length(still_missing) > 0) { \
                  cat('WARNING: Could not install Bioconductor packages:', \
                      paste(still_missing, collapse=', '), '\n'); \
                  cat('Build will continue; verification step will handle fallback.\n') \
              } else { \
                  cat('All', length(bioc_pkgs), 'Bioconductor packages installed OK\n') \
              }"; \
    fi

# Install all CRAN dependencies (pre-compiled binaries from RSPM)
# NMF depends on Biobase (Bioconductor), so Bioconductor must be installed first
# Use dependencies = NA (Depends + Imports + LinkingTo only, not Suggests)
# to avoid pulling hundreds of optional packages that waste disk space
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing CRAN packages (binary)..." && \
        echo "========================================" && \
        R -e "options(timeout = 600, \
                  repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'))); \
              install.packages(c( \
                  'devtools', \
                  'Matrix', 'Rcpp', 'RcppArmadillo', 'RcppEigen', 'RcppParallel', \
                  'ggplot2', 'dplyr', 'tidyr', 'data.table', \
                  'httr', 'jsonlite', 'R6', 'crayon', \
                  'reshape2', 'patchwork', 'NMF', 'akima', \
                  'gganimate', 'metap', 'circlize', 'ggalluvial', \
                  'networkD3', 'survival', 'survminer', 'ggpubr', \
                  'car', 'lme4', 'sp', \
                  'scatterpie', 'png', 'shiny', 'plotly', 'DT', \
                  'factoextra', 'NbClust', 'cluster', 'pbmcapply', \
                  'psych', 'RANN', 'sctransform', \
                  'irlba', 'igraph', 'Rtsne', 'ROCR', 'entropy' \
              ), dependencies = NA, Ncpus = parallel::detectCores())"; \
    fi

# Install arrow separately (large package with C++ library download)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing arrow package..." && \
        echo "========================================" && \
        R -e "options(timeout = 600, \
                  repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'))); \
              tryCatch({ \
                  install.packages('arrow', dependencies = NA, \
                      Ncpus = parallel::detectCores()); \
                  library(arrow); \
                  cat('arrow', as.character(packageVersion('arrow')), 'OK\n') \
              }, error = function(e) { \
                  cat('WARNING: arrow install failed:', conditionMessage(e), '\n'); \
                  cat('Retrying with LIBARROW_BINARY=TRUE...\n'); \
                  Sys.setenv(LIBARROW_BINARY = 'TRUE', NOT_CRAN = 'TRUE'); \
                  install.packages('arrow', dependencies = NA, \
                      Ncpus = parallel::detectCores()); \
                  library(arrow); \
                  cat('arrow', as.character(packageVersion('arrow')), 'OK (retry)\n') \
              })"; \
    fi

# Verify all pre-installed R dependencies are present
# install.packages() does NOT error on individual failures, so check explicitly
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Verifying pre-installed R dependencies..." && \
        echo "========================================" && \
        R -e "required <- c( \
                  'Matrix', 'ggplot2', 'reshape2', 'patchwork', 'NMF', 'akima', \
                  'gganimate', 'metap', 'circlize', 'ggalluvial', 'networkD3', \
                  'survival', 'survminer', 'ComplexHeatmap', \
                  'RcppParallel', \
                  'jsonlite', 'scatterpie', 'png', 'shiny', 'plotly', 'DT', \
                  'factoextra', 'NbClust', 'cluster', 'pbmcapply', 'psych', \
                  'arrow', 'RANN', 'sctransform', 'UCell', 'BiRewire', 'limma', \
                  'sva', 'irlba', 'igraph', 'Rtsne', 'ROCR', 'entropy' \
              ); \
              cat('Checking', length(required), 'required packages...\n'); \
              status <- sapply(required, requireNamespace, quietly = TRUE); \
              for (i in seq_along(required)) { \
                  cat(sprintf('  %-20s %s\n', required[i], \
                      if (status[i]) 'OK' else 'MISSING')) \
              }; \
              missing <- required[!status]; \
              if (length(missing) > 0) { \
                  cat('\nWARNING:', length(missing), 'packages missing after initial install:\n'); \
                  cat('  ', paste(missing, collapse=', '), '\n'); \
                  cat('Attempting fallback install via BiocManager...\n'); \
                  options(timeout = 600, \
                      repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'))); \
                  for (pkg in missing) { \
                      cat('  Installing', pkg, '...\n'); \
                      tryCatch( \
                          BiocManager::install(pkg, ask = FALSE, update = FALSE, \
                              Ncpus = parallel::detectCores()), \
                          error = function(e) cat('    FAILED:', conditionMessage(e), '\n') \
                      ) \
                  } \
              }; \
              still_missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]; \
              if (length(still_missing) > 0) { \
                  cat('\n========================================\n'); \
                  cat('FATAL: Cannot install required packages:\n'); \
                  for (pkg in still_missing) cat('  -', pkg, '\n'); \
                  cat('========================================\n'); \
                  cat('System info: R', R.version.string, '\n'); \
                  cat('BiocManager version:', as.character(packageVersion('BiocManager')), '\n'); \
                  tryCatch(cat('Bioconductor version:', \
                      as.character(BiocManager::version()), '\n'), \
                      error = function(e) NULL); \
                  stop(paste('FATAL: Cannot install required packages:', \
                      paste(still_missing, collapse=', '))) \
              } else { \
                  cat('\nAll', length(required), 'required R packages verified OK\n') \
              }"; \
    fi

# Install GitHub R packages
# GITHUB_PAT increases API rate limit from 60 to 5000 per hour
# Fallback to dependencies=NA if dependencies=FALSE fails
ARG GITHUB_PAT
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing MUDAN from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              tryCatch({ \
                  remotes::install_github('JEFworks/MUDAN', \
                      dependencies = FALSE, force = TRUE); \
                  library(MUDAN); \
                  cat('MUDAN', as.character(packageVersion('MUDAN')), 'OK\n') \
              }, error = function(e) { \
                  message('First attempt failed: ', conditionMessage(e)); \
                  message('Retrying with dependencies...'); \
                  remotes::install_github('JEFworks/MUDAN', \
                      dependencies = NA, force = TRUE); \
                  library(MUDAN); \
                  cat('MUDAN', as.character(packageVersion('MUDAN')), 'OK (with deps)\n') \
              })"; \
    fi

# Install RidgeR from GitHub (beibeiru/RidgeR)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing RidgeR from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              tryCatch({ \
                  remotes::install_github('beibeiru/RidgeR', \
                      dependencies = FALSE, force = TRUE); \
                  library(RidgeR); \
                  cat('RidgeR', as.character(packageVersion('RidgeR')), 'OK\n') \
              }, error = function(e) { \
                  message('First attempt failed: ', conditionMessage(e)); \
                  message('Retrying with dependencies...'); \
                  remotes::install_github('beibeiru/RidgeR', \
                      dependencies = NA, force = TRUE); \
                  library(RidgeR); \
                  cat('RidgeR', as.character(packageVersion('RidgeR')), 'OK (with deps)\n') \
              })"; \
    fi

# Install SecAct from GitHub (data2intelligence/SecAct)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing SecAct from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              tryCatch({ \
                  remotes::install_github('data2intelligence/SecAct', \
                      dependencies = FALSE, force = TRUE); \
                  library(SecAct); \
                  cat('SecAct', as.character(packageVersion('SecAct')), 'OK\n') \
              }, error = function(e) { \
                  message('First attempt failed: ', conditionMessage(e)); \
                  message('Retrying with dependencies...'); \
                  remotes::install_github('data2intelligence/SecAct', \
                      dependencies = NA, force = TRUE); \
                  library(SecAct); \
                  cat('SecAct', as.character(packageVersion('SecAct')), 'OK (with deps)\n') \
              })"; \
    fi

# Install SpaCET from GitHub (data2intelligence/SpaCET)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing SpaCET from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              tryCatch({ \
                  remotes::install_github('data2intelligence/SpaCET', \
                      dependencies = FALSE, force = TRUE); \
                  library(SpaCET); \
                  cat('SpaCET', as.character(packageVersion('SpaCET')), 'OK\n') \
              }, error = function(e) { \
                  message('First attempt failed: ', conditionMessage(e)); \
                  message('Retrying with dependencies...'); \
                  remotes::install_github('data2intelligence/SpaCET', \
                      dependencies = NA, force = TRUE); \
                  library(SpaCET); \
                  cat('SpaCET', as.character(packageVersion('SpaCET')), 'OK (with deps)\n') \
              })"; \
    fi

# Verify R installation (fail build if required packages are missing)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Verifying R installation..." && \
        echo "========================================" && \
        R -e "cat('R version:', R.version.string, '\n'); \
              required <- c('RidgeR', 'SecAct', 'SpaCET'); \
              all_ok <- TRUE; \
              for (pkg in required) { \
                  if (requireNamespace(pkg, quietly = TRUE)) { \
                      cat(pkg, as.character(packageVersion(pkg)), 'OK\n') \
                  } else { \
                      cat(pkg, 'MISSING\n'); \
                      all_ok <- FALSE \
                  } \
              }; \
              if (!all_ok) stop('Required R packages are missing!')"; \
    fi

# =============================================================================
# Python: Install SecActPy
# =============================================================================

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install base Python packages (pin numpy<2.0.0 to avoid ABI breaking changes)
RUN pip3 install --no-cache-dir \
    "numpy>=1.20.0,<2.0.0" \
    pandas \
    scipy \
    h5py \
    anndata \
    scanpy \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab

# Install CuPy for GPU version only
ARG USE_GPU
RUN if [ "$USE_GPU" = "true" ]; then \
        echo "Installing CuPy for GPU support..." && \
        pip3 install --no-cache-dir cupy-cuda11x; \
    else \
        echo "Skipping CuPy (CPU-only build)"; \
    fi

# Install SecActPy from official GitHub repository (always latest version)
RUN pip3 install --no-cache-dir git+https://github.com/data2intelligence/SecActpy.git

# Verify Python installation
RUN python3 -c "import secactpy; print(f'SecActPy {secactpy.__version__} OK, GPU: {secactpy.CUPY_AVAILABLE}')"

# =============================================================================
# Environment
# =============================================================================

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GPU environment variables (harmless if not using GPU)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Entry Point
# =============================================================================

CMD ["/bin/bash"]

# =============================================================================
# Labels
# =============================================================================

LABEL maintainer="Seongyong Park <https://github.com/psychemistz>"
LABEL description="SecActPy - Secreted Protein Activity Inference (CPU/GPU)"
LABEL version="0.2.1"
LABEL org.opencontainers.image.source="https://github.com/data2intelligence/SecActpy"
