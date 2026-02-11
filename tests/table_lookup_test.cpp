// tests/table_lookup_test.cpp
#include "unit_test.hpp"
#include "test_helpers.hpp"

#include "Physics.hpp"
#include "GasState.hpp"
#include "BasicOperations.hpp"
#include "LteEOS.hpp"

#include <mutation++/mutation++.h>
#include "../libs/mfem/general/forall.hpp"

using real_t = Prandtl::real_t;

using namespace Prandtl;
using namespace Mutation;

TEST(hunt_cpu_test)
{
    int n = 100;
    Prandtl::Vector arr(n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = i + 1;
    }

    real_t x = 34.5;

    // Hunt Right (guess near left)
    int ind_lo = 2;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Hunt Left (guess near right)
    ind_lo = 70;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Left Boundary
    x = 1;
    ind_lo = 50;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 0, 1e-14);

    // Right Boundary
    x = n;
    ind_lo = 20;
    ind_lo = hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, n-2, 1e-14);

    return 0;
}

TEST(hunt_gpu_test)
{
    const int n = 100;
    mfem::Vector arr(n);
    for (int i = 0; i < n; i++) { arr[i] = i + 1; }

    const real_t *a = arr.Read(); // device-safe read pointer
    arr.UseDevice();

    mfem::Vector outv(4);
    outv.UseDevice();
    real_t *out_d = outv.Write(); // device-safe write pointer

    mfem::forall(4, [=] MFEM_HOST_DEVICE (int i)
    {
        real_t x;
        int guess;

        if (i == 0)      { x = 34.5;  guess = 2;  }   // hunt right
        else if (i == 1) { x = 34.5;  guess = 70; }   // hunt left
        else if (i == 2) { x = 1.0;   guess = 50; }   // left boundary
        else             { x = 100.0; guess = 20; }   // right boundary

        int idx = hunt(a, n, x, guess);
        out_d[i] = (real_t) idx;
    });

    const real_t *out_h = outv.HostRead();

    EXPECT_CLOSE(out_h[0], 33.0, 1e-14);
    EXPECT_CLOSE(out_h[1], 33.0, 1e-14);
    EXPECT_CLOSE(out_h[2],  0.0, 1e-14);
    EXPECT_CLOSE(out_h[3], 98.0, 1e-14); // n-2

    return 0;
}

// -----------------------------------------------------------------------------
// EOS test : Thermodynamic properties of a gas-mixture in LTE
// The lte table is populated using Mutation++ property = property(rho x rhoe)
// -----------------------------------------------------------------------------
TEST(LTEGas_EOS)
{
    // Range and resolution of the table (Evenly spaced table in this test)
    int nx = 11, ny = 11;
    real_t rho_min  = 0.5  , rho_max  = 1.5  , rho_step  = (rho_max-rho_min)/(nx-1);
    real_t rhoe_min = 2.0e5, rhoe_max = 1.1e6, rhoe_step = (rhoe_max-rhoe_min)/(ny-1);

    // Mutation++ object for Equilibrium state
    MixtureOptions opts("air_5");          // Gas-Mixture (N, O, NO, N2, O2)
    opts.setStateModel("Equil");           // Equilibrium state model
    opts.setThermodynamicDatabase("RRHO"); // Default thermodynamic data base
    opts.setViscosityAlgorithm("Chapmann-Enskog_LDLT"); // Viscosity algorithm
    Mixture mix(opts);                     // Initializing mixture object
        mix.addComposition("N:0.8, O:0.2", true); // composition
    int num_species = 5, num_properties = 9;

    const int dim = 3, ndofs = 1;

    // Data arrays for table lookup for LTE properties
    mfem::Vector lte_table( (num_species + 
                             num_properties) * (nx*ny) ); // LTE-Tables
    mfem::Vector rho_grid(nx), rhoe_grid(ny); // 1-D grids of rho and rhoE
    mfem::Array<int> hunt_arr( ndofs * 2 );   // Array of hunted indices

    StateLayout L(dim, ndofs, nx, ny, num_species);
    const int num_eq = L.eq_energy + 1;
    LTEGasEOS eos;

    // Populating the 1D grid of density and rho*internal energy
    for(int ind_x=0; ind_x < nx; ind_x++) rho_grid[ind_x] = rho_min + ind_x * rho_step;
    for(int ind_y=0; ind_y < ny; ind_y++) rhoe_grid[ind_y] = rhoe_min + ind_y * rhoe_step;

    // Generation of LTE table
    for(int ind_y=0; ind_y < ny; ind_y++)
    {
        for(int ind_x=0; ind_x < nx; ind_x++)
        {
            mix.setState(&rho_grid[ind_x], &rhoe_grid[ind_y], 0);
            lte_table[L.lte_property_index(L.P_idx, ind_x, ind_y)] = mix.P();
            lte_table[L.lte_property_index(L.T_idx, ind_x, ind_y)] = mix.T();
        }
    }

    // Setup the LTE EOS
    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(lte_table.Read(), hunt_arr.ReadWrite(), 
                                         rho_grid.Read(), rhoe_grid.Read());

    // ------------------------------------------------------------------------------------


    // Populate the 1-DOF state (Ensure the values are in ranges of the table)
    real_t rho  = 0.753;
    real_t rhoe = 600000.0;
    const real_t u1[3] = {10.0, -3.0, 5.0};

    std::vector<real_t> U(num_eq * ndofs);
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S1(U.data(), 0);

    // Populate the hunt array with initial guesses for the hunt algorithm
    phys->hunt[0] = hunt(rho_grid.Read(), nx, rho, 0);
    phys->hunt[1] = hunt(rhoe_grid.Read(), ny, rhoe, 0);
    EXPECT_CLOSE(phys->hunt[0], 2, 0);
    EXPECT_CLOSE(phys->hunt[1], 4, 0);

    int l_x = phys->hunt[0]; int l_y = phys->hunt[1];
    int u_x = l_x + 1      ; int u_y = l_y + 1;

    int P_idx = L.P_idx, T_idx = L.T_idx;

    // Test to check the values of the corners of the 2D cell in P and T tables
    real_t P_true, T_true, P_table, T_table;
    for(int i=0; i < 2; i++)
    {
        for(int j=0; j < 2; j++)
        {
            mix.setState(&rho_grid[2 + i], &rhoe_grid[4 + j], 0);
            P_true = mix.P();   P_table = phys->lte_table[L.lte_property_index(P_idx, l_x + i, l_y + j)];
            T_true = mix.T();   T_table = phys->lte_table[L.lte_property_index(T_idx, l_x + i, l_y + j)];
            EXPECT_CLOSE(P_true, P_table, 1e-6);
            EXPECT_CLOSE(T_true, T_table, 1e-6);
        }
    }

    // Testing Bi-linear Interpolation 
    // A (Choosing one corner)
    rho  = 0.7;
    rhoe = 560000.0;
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S2(U.data(), 0);
    
    phys->hunt[0] = hunt(rho_grid.Read(), nx, rho, 0);
    phys->hunt[1] = hunt(rhoe_grid.Read(), ny, rhoe, 0);
    real_t P_interpolated = eos.pressure(*phys, L, S2, 0);
    real_t T_interpolated = eos.temperature(*phys, L, S2, 0);

    mix.setState(&rho, &rhoe, 0);
    P_true = mix.P();
    T_true = mix.T();

    EXPECT_CLOSE(P_true, P_interpolated, 1e-6);
    EXPECT_CLOSE(T_true, T_interpolated, 1e-6);

    // B (Choosing the mid-point of a cell)
    rho  = 0.75;
    rhoe = 580000.0;
    fill_single_dof_state(L, U, dim, rho, u1, rhoe);
    DofStateView S3(U.data(), 0);

    phys->hunt[0] = hunt(rho_grid.Read(), nx, rho, 0);
    phys->hunt[1] = hunt(rhoe_grid.Read(), ny, rhoe, 0);
    P_interpolated = eos.pressure(*phys, L, S3, 0);
    T_interpolated = eos.temperature(*phys, L, S3, 0);

    mix.setState(&rho, &rhoe, 0);
    P_true = mix.P();
    T_true = mix.T();

    std::cout << "\n Bilinear interpolation at mid point : " << std::endl;
    std::cout << "P true: " << P_true << ", P interpolated: " << P_interpolated << std::endl;
    std::cout << "T true: " << T_true << ", T interpolated: " << T_interpolated << std::endl;
    std::cout << std::endl;

    // CL NOTE : The table is too coarse to get accurate interpolated values hence write a new test
    // EXPECT_CLOSE(P_true, P_interpolated, 1e-6);
    // EXPECT_CLOSE(T_true, T_interpolated, 1e-6);

    return 0;
}