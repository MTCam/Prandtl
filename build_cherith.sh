rm -rf build-euler

mkdir -p build-euler
MFEM_DIR="libs/mfem/build"
cmake -S . -B build-euler -G Ninja \
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
            -DBUILD_TESTING=ON \
            -DCMAKE_PREFIX_PATH="${MFEM_DIR}" \
            -DPARABOLIC=OFF -DSUBCELL_FV_BLENDING=ON -DLTE_EOS=ON

cmake --build build-euler -- -k 0
