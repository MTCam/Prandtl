rm -rf build-cns

mkdir -p build-cns

cmake -S . -B build-cns -G Ninja \
            -DCONFIG_FILE=compile_config.json \
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
            -DBUILD_TESTING=ON \
            -DMFEM_DIR=libs/mfem \
            -DHYPRE_DIR=libs/hypre/src/hypre \
            -DMETIS_DIR=libs/metis-5.1.0/ \
            -Dmutation++_DIR="$PWD/libs/Mutationpp/install/lib/cmake/mutation++" \
            -DPARABOLIC=ON -DSUBCELL_FV_BLENDING=ON

cmake --build build-cns -- -k 0
