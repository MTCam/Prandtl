#!/bin/bash

# This script performs a clean build of all Prandtl dependencies.

# Exit immediately if any command fails.
set -e

echo "--- Starting dependencies build ---"
PROJECT_ROOT=$(pwd)
IS_HPC=false

# --- Environment Detection ---
if command -v module &> /dev/null; then
    IS_HPC=true
fi

# --- Step 1: Build HYPRE ---
echo "--- Building HYPRE ---"
cd libs/hypre/src/
./configure --disable-fortran
make clean
make -j4
cd ../../../

# --- Step 2: Build METIS 5 ---
echo "--- Building METIS ---"
cd libs/metis-5.1.0/
make distclean
make BUILDDIR=lib config
make BUILDDIR=lib
cp lib/libmetis/libmetis.a lib
cd ../../

# --- Step 3: Build Parallel MFEM ---
echo "--- Building Parallel MFEM ---"
cd libs/mfem/
make distclean
make parallel -j4 MFEM_USE_METIS_5=YES METIS_DIR="$(cd ../metis-5.1.0/ && pwd)" HYPRE_DIR="$(cd ../hypre/src/hypre && pwd)"
cd ../../

# --- Step 4: Build GLVis ---
if [ "$IS_HPC" = false ]; then
echo "--- Building GLVis ---"
cd glvis/
make distclean
make MFEM_DIR="$(cd ../libs/mfem && pwd)" -j4
cd ../
fi

echo "--- All local dependencies built successfully! ---"
