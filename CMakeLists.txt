cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Prandtl)

option(BUILD_TESTING "Build unit tests" ON)
option(PARABOLIC "Enable Compressible Navier-Stokes Solver" OFF)
option(SUBCELL_FV_BLENDING "Enable subcell blending" ON)
option(AXISYMMETRIC "Enable Axisymmetric domain solver" OFF)
option(SUTHERLAND "Enable Sutherland transport model" OFF)
option(LTE_EOS "Enable LTE EOS" OFF)
option(ENABLE_TIMERS "Enable performance timers." OFF)

find_package(MFEM REQUIRED CONFIG) 
if (TARGET MFEM::mfem)
  set(PRANDTL_MFEM_TARGET MFEM::mfem)
elseif (TARGET mfem)
  set(PRANDTL_MFEM_TARGET mfem)
else()
  message(FATAL_ERROR "MFEM package found, but no MFEM target (MFEM::mfem or mfem) was provided.")
endif()

find_package(MPI REQUIRED)
if (LTE_EOS)
  message(STATUS, "Building Prandtl with LTE EOS")
  find_package(mutation++ REQUIRED)
  find_package(Eigen3 REQUIRED)
endif()

add_executable(${PROJECT_NAME} main.cpp)

# --- Python Script for Compile Definitions ---
set(CONFIG_FILE "${CONFIG_FILE}" CACHE STRING "Path to config file")
if(IS_ABSOLUTE "${CONFIG_FILE}")
    set(FULL_CONFIG_PATH "${CONFIG_FILE}")
else()
    set(FULL_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_FILE}")
endif()
message(STATUS "CONFIG_FILE = ${CONFIG_FILE}")
if(NOT "${CONFIG_FILE}" STREQUAL "")
    message(STATUS "Parsing configuration file: ${FULL_CONFIG_PATH}")
    execute_process(
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/parse_compile_config.py" "${FULL_CONFIG_PATH}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE COMPILE_DEFINES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )       
    string(REPLACE "\n" ";" COMPILE_DEFINES_LIST "${COMPILE_DEFINES}")
    message(STATUS "COMPILE_DEFINES_LIST = ${COMPILE_DEFINES_LIST}")

    target_compile_definitions(${PROJECT_NAME}
        PRIVATE ${COMPILE_DEFINES_LIST}
    )
else()
    message(STATUS "No configuration file given.")
endif()

if(ENABLE_TIMERS)
  message(STATUS "Enabling performance timers.")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TIMERS)
endif()

if(PARABOLIC)
  message(STATUS "Building Prandtl for: Compressible Navier-Stokes equations")
  target_compile_definitions(${PROJECT_NAME} PRIVATE PARABOLIC)
else()
  message(STATUS "Building Prandtl for: Euler equations")
endif()

if(SUBCELL_FV_BLENDING)
  message(STATUS "Building Prandtl with: Subcell FV Blending enabled")
  target_compile_definitions(${PROJECT_NAME} PRIVATE SUBCELL_FV_BLENDING)
else()
  message(STATUS "Building Prandtl with: Subcell FV Blending OFF")
endif()

if(SUTHERLAND)
  message(STATUS "Building Prandtl with: Sutherland transport model enabled")
  target_compile_definitions(${PROJECT_NAME} PRIVATE SUTHERLAND)
endif()

if(LTE_EOS)
  message(STATUS "Building Prandtl with: LTE EOS enabled")
  target_compile_definitions(${PROJECT_NAME} PRIVATE LTE_EOS)
endif()

if(AXISYMMETRIC)
  message(STATUS "Building Prandtl with: Axisymmetric domain enabled")
  target_compile_definitions(${PROJECT_NAME} PRIVATE AXISYMMETRIC)
endif()

# -------------------------------------------------------------------------------
# Target-local sanitizers for a single executable (Clang/AppleClang/GNU)
# -------------------------------------------------------------------------------

option(ENABLE_ASAN_UBSAN "Enable ASan+UBSan for the main executable" OFF)
option(NO_OPT "Disable optimizations for testing" OFF)


target_compile_definitions(${PROJECT_NAME} PRIVATE
  "$<$<OR:$<CONFIG:Debug>,$<BOOL:${NO_OPT}>,$<BOOL:${ENABLE_ASAN_UBSAN}>>:MFEM_DEBUG>"
)

if (ENABLE_ASAN_UBSAN AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  message(STATUS "Enabling ASan+UBSan on target: ${PROJECT_NAME}")

  # Compile flags
  target_compile_options(${PROJECT_NAME} PRIVATE
    -O1
    -g
    -fno-fast-math
    -ffp-contract=off
    -fno-unsafe-math-optimizations
    -fno-associative-math
    -fno-reciprocal-math
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
    -fsanitize=address,undefined
  )

  # Link flags
  target_link_options(${PROJECT_NAME} PRIVATE
    -fsanitize=address,undefined
  )

  # Prefer hard-fail on first UB to get the cleanest stack trace
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-sanitize-recover=all)
  endif()
endif()

if (NO_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  message(STATUS "Disabling all optimizations on target: ${PROJECT_NAME}")

  # Compile flags
  target_compile_options(${PROJECT_NAME} PRIVATE
    -O0
    -g
    -fno-fast-math
    -ffp-contract=off
    -fno-unsafe-math-optimizations
    -fno-associative-math
    -fno-reciprocal-math
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
  )

endif()

target_sources(${PROJECT_NAME}
    PRIVATE src/Operators/DGSEMOperator.cpp 
    PRIVATE src/Integrators/DGSEMIntegrator/DGSEMIntegrator.cpp
    PRIVATE src/Integrators/Lifting/LiftingScheme.cpp
    PRIVATE src/Integrators/Lifting/LiftingBR1/LiftingBR1.cpp
    PRIVATE src/RiemannSolvers/RoeFlux.cpp
    PRIVATE src/RiemannSolvers/LaxFriedrichsFlux.cpp
    PRIVATE src/RiemannSolvers/HLLFlux.cpp
    PRIVATE src/RiemannSolvers/ChandrashekarFlux.cpp
    PRIVATE src/BasicOperations/BasicOperations.cpp
    PRIVATE src/Filters/EntropyFilter.cpp
    PRIVATE src/Integrators/BoundaryConditions/BdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/NoSlipAdiabWallBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/NoSlipIsothWallBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SpecifiedStateBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SlipWallBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/RiemannInvariantBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SubsonicInflowPtTtAngBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SubsonicInflowRVBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SubsonicOutflowPBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SupersonicInflowBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SupersonicOutflowBdrFaceIntegrator.cpp
    PRIVATE src/Integrators/BoundaryConditions/SymmetryBdrFaceIntegrator.cpp
    PRIVATE src/Fluxes/NavierStokesFlux.cpp
    PRIVATE src/ModalBasis/ModalBasis.cpp
    PRIVATE src/Limiters/MinmodLimiter.cpp
    PRIVATE src/Indicators/Indicator.cpp
    PRIVATE src/Indicators/PerssonPeraireIndicator.cpp
    PRIVATE src/Simulation/Simulation.cpp
    PRIVATE src/Operators/NonlinearForm/DGSEMNonlinearForm.cpp
    )

target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Device
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Operators/NonlinearForm
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Operators
    PUBLIC ${CMAKE_SOURCE_DIR}/src/RiemannSolvers
    PUBLIC ${CMAKE_SOURCE_DIR}/src/BasicOperations
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Filters
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Integrators/BoundaryConditions
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Integrators/Lifting/LiftingBR1
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Integrators/Lifting
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Integrators/DGSEMIntegrator
    PUBLIC ${CMAKE_SOURCE_DIR}/src/ModalBasis
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Physics
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Limiters
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Indicators
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Simulation
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Include
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Fluxes
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/NavierStokes/2D/LidDrivenCavity
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/NavierStokes/2D/TaylorGreenVortex
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/NavierStokes/3D/TaylorGreenVortex
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/SodShockTube
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/ModifiedSodShockTube
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/LaxShockTube
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/LeBlancShockTube
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/Problem123
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/WoodwardColellaBlastWave
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/WoodwardColellaBlastWaveLeft
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/WoodwardColellaBlastWaveRight
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/WoodwardColellaBlastWaveCollision
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/ShuOsherShock
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/1D/AcousticWave
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/DoubleMachReflection
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/ForwardFacingStep
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/BackwardFacingStep
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/TriplePointShockInteraction
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/KelvinHelmholtzInstability
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/IsentropicVortex
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/Ramp
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/Euler/2D/Nagashima_Ramjet
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/LTE/Euler/LTEVortex
    )

if (LTE_EOS)
  target_link_libraries(${PROJECT_NAME}
      PRIVATE
      ${PRANDTL_MFEM_TARGET}
      MPI::MPI_CXX
      mutation++::mutation++
      Eigen3::Eigen
  )
else()
  target_link_libraries(${PROJECT_NAME}
      PRIVATE
      ${PRANDTL_MFEM_TARGET}
      MPI::MPI_CXX
  )
endif()


if (BUILD_TESTING)
  include(CTest)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()
