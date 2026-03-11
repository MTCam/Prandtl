rm -rf build-cns

mkdir -p build-cns
MFEM_DIR="libs/mfem/build"
Mutationpp_DIR="libs/Mutationpp/install"
Eigen3_DIR="/usr/include/eigen3"
cmake -S . -B build-cns -G Ninja \
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
            -DBUILD_TESTING=ON \
            -DCMAKE_PREFIX_PATH="${MFEM_DIR};${Mutationpp_DIR};${Eigen3_DIR}" \
            -DPARABOLIC=ON -DSUBCELL_FV_BLENDING=ON

cmake --build build-cns -- -k 0
