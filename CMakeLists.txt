cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_COMPILER ${MPICXX})
# set(MPI_CXX_INCLUDE_PATH "")
# set(MPI_CXX_LIBRARIES "")

project(Prandtl)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES})

message(STATUS "MPI CXX COMPILER: ${MPI_CXX_COMPILER}")
message(STATUS "MPI INCLUDE PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI LIBRARIES: ${MPI_CXX_LIBRARIES}")

# Set the CONFIG_FILE variable, with a default if not provided
set(CONFIG_FILE "${CONFIG_FILE}" CACHE STRING "Path to config file")

message(STATUS "CONFIG_FILE = ${CONFIG_FILE}")

# --- Python Script for Compile Definitions ---
set(CONFIG_FILE "${CONFIG_FILE}" CACHE STRING "Path to config file")
message(STATUS "CONFIG_FILE = ${CONFIG_FILE}")

if(IS_ABSOLUTE "${CONFIG_FILE}")
    set(FULL_CONFIG_PATH "${CONFIG_FILE}")
else()
    set(FULL_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_FILE}")
endif()

execute_process(
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/parse_compile_config.py"
                    "${FULL_CONFIG_PATH}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE COMPILE_DEFINES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" COMPILE_DEFINES_LIST "${COMPILE_DEFINES}")
message(STATUS "COMPILE_DEFINES_LIST = ${COMPILE_DEFINES_LIST}")

add_executable(${PROJECT_NAME} main.cpp)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE ${COMPILE_DEFINES_LIST}
)

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
    PUBLIC ${CMAKE_SOURCE_DIR}/libs/mfem
    PUBLIC ${CMAKE_SOURCE_DIR}/libs/hypre/src/hypre/include
    PUBLIC ${CMAKE_SOURCE_DIR}/libs/metis-5.1.0/include
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
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Device
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Limiters
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Indicators
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Simulation
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Include
    PUBLIC ${CMAKE_SOURCE_DIR}/src/Fluxes
    PUBLIC ${CMAKE_SOURCE_DIR}/TestCases/NavierStokes/2D/LidDrivenCavity
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
    )

target_link_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_SOURCE_DIR}/libs/mfem
    PUBLIC ${CMAKE_SOURCE_DIR}/libs/hypre/src/lib
    PUBLIC ${CMAKE_SOURCE_DIR}/libs/metis-5.1.0/lib
    )

target_link_libraries(${PROJECT_NAME} mfem HYPRE metis MPI::MPI_CXX)


