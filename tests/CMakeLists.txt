# This file is part of the unit testing infrastructure for Prandtl
#
# Example on how to add unit tests
# add_executable(suite_test_exe
#     unit_test_main.cpp
#     suite_tests.cpp
# )
#
# Run the whole suite of tests like this:
# add_test(NAME test_suite_name COMMAND suite_test_exe)
# Select a single test from the suite like this:
# add_test(NAME suite::test1 COMMAND suite_test_exe test1)
#

# Unit testing self-test suite
add_executable(unit_testing_stub
    unit_test_main.cpp
    test_stub.cpp
)

add_test(NAME UnitTestingInfrastructure COMMAND unit_testing_stub)
add_test(NAME SingleUnitTestInfrastructure COMMAND unit_testing_stub unit_testing_funcs)

# Gas Physics testing suite
add_executable(gas_tests
   unit_test_main.cpp
   eos_tests.cpp
   transport_tests.cpp
   gas_model_tests.cpp
)

target_include_directories(gas_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Physics
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Device
)

# Propagate SUTHERLAND compile definition to gas_tests if enabled
if(SUTHERLAND)
    target_compile_definitions(gas_tests PRIVATE SUTHERLAND)
endif()

target_link_libraries(gas_tests PRIVATE ${PRANDTL_MFEM_TARGET})

add_test(NAME GasPhysicsTestingSuite COMMAND gas_tests)

# State unit testing suite
add_executable(state_tests
   unit_test_main.cpp
   state_tests.cpp
)

target_include_directories(state_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Physics
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Device
)

target_link_libraries(state_tests PRIVATE ${PRANDTL_MFEM_TARGET})

add_test(NAME StateTestingSuite COMMAND state_tests)
    
# Table lookup unit testing suite
add_executable(table_lookup_tests
    unit_test_main.cpp
    ${PROJECT_SOURCE_DIR}/src/BasicOperations/BasicOperations.cpp
    table_lookup_test.cpp
)

# Mutation++ local install prefix
set(MUTATIONPP_PREFIX ${PROJECT_SOURCE_DIR}/libs/Mutationpp/install)
set(MUTATIONPP_DATA   ${PROJECT_SOURCE_DIR}/libs/Mutationpp/data)

target_include_directories(table_lookup_tests
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/mfem
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/hypre/src/hypre/include
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/metis-5.1.0/include
    ${PROJECT_SOURCE_DIR}/src/BasicOperations
    ${PROJECT_SOURCE_DIR}/src/Physics
    ${PROJECT_SOURCE_DIR}/src/Device
    PUBLIC ${MUTATIONPP_PREFIX}/include
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/Mutationpp/data
    PUBLIC ${EIGEN3_INCLUDE_DIR}
)

target_link_directories(table_lookup_tests
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/mfem
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/hypre/src/lib
    PUBLIC ${PROJECT_SOURCE_DIR}/libs/metis-5.1.0/lib
    PUBLIC ${MUTATIONPP_PREFIX}/lib
)

target_link_libraries(table_lookup_tests
    mfem HYPRE metis MPI::MPI_CXX
    mutation++ Eigen3::Eigen
)

add_test(NAME TableLookupTestingSuite COMMAND table_lookup_tests)
set_tests_properties(TableLookupTestingSuite PROPERTIES
  ENVIRONMENT "MPP_DATA_DIRECTORY=${MUTATIONPP_DATA}"
)
