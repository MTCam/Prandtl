// tests/state_tests.cpp
#include "unit_test.hpp"
#include "state_semantics.hpp"
#include "legacy_state_adapter.hpp"
#include "GasState.hpp"
#include "gas_state_adapter.hpp"

using real_t = Prandtl::real_t;

TEST(LegacyState_MassMomentumEnergy_2D)
{
    const int dim   = 2;   // or test dim=3 in another test
    const int ndofs = 5;   // small number is fine

    LegacyConservativeState state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(LegacyState_MassMomentumEnergy_3D)
{
    const int dim   = 3;   // or test dim=3 in another test
    const int ndofs = 5;   // small number is fine

    LegacyConservativeState state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(GasState_MassMomentumEnergy_2D)
{
    const int dim   = 2;
    const int ndofs = 5;

    GasStateSemanticsAdapter state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(GasState_MassMomentumEnergy_3D)
{
    const int dim   = 3;
    const int ndofs = 5;

    GasStateSemanticsAdapter state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}

TEST(StateLayout_Indexing_NoScalars_2D)
{
    const int dim   = 2;
    const int ndofs = 5;

    Prandtl::StateLayout layout(dim, ndofs);

    // Basic metadata
    EXPECT_CLOSE(layout.dim,             dim,     0.0);
    EXPECT_CLOSE(layout.num_dofs_scalar, ndofs,   0.0);
    EXPECT_CLOSE(layout.eq_mass,         0,       0.0);
    EXPECT_CLOSE(layout.eq_mom[0],       1,       0.0);
    EXPECT_CLOSE(layout.eq_mom[1],       2,       0.0);
    EXPECT_CLOSE(layout.eq_mom[2],      -1,       0.0); // unused in 2D
    EXPECT_CLOSE(layout.eq_energy,       dim+1,   0.0);
    EXPECT_CLOSE(layout.eq_scalar0,     -1,       0.0);
    EXPECT_CLOSE(layout.num_scalars,     0,       0.0);

    // Flat index should be eq * ndofs + dof
    const int num_eq = dim + 2; // rho + dim momenta + energy
    for (int eq = 0; eq < num_eq; ++eq)
    {
        for (int i = 0; i < ndofs; ++i)
        {
            const int expected = eq * ndofs + i;
            EXPECT_CLOSE(layout.index(eq, i), expected, 0.0);
        }
    }

    return 0;
}

TEST(StateLayout_Indexing_WithScalars_3D)
{
    const int dim        = 3;
    const int ndofs      = 4;
    const int num_scalars = 2;

    Prandtl::StateLayout layout(dim, ndofs, num_scalars);

    EXPECT_CLOSE(layout.dim,             dim,                  0.0);
    EXPECT_CLOSE(layout.num_dofs_scalar, ndofs,                0.0);
    EXPECT_CLOSE(layout.eq_mass,         0,                    0.0);
    EXPECT_CLOSE(layout.eq_mom[0],       1,                    0.0);
    EXPECT_CLOSE(layout.eq_mom[1],       2,                    0.0);
    EXPECT_CLOSE(layout.eq_mom[2],       3,                    0.0);
    EXPECT_CLOSE(layout.eq_energy,       dim + 1,              0.0); // 4
    EXPECT_CLOSE(layout.eq_scalar0,      dim + 2,              0.0); // 5
    EXPECT_CLOSE(layout.num_scalars,     num_scalars,          0.0);

    const int num_eq = dim + 2 + num_scalars; // rho, 3 mom, E, 2 scalars

    for (int eq = 0; eq < num_eq; ++eq)
    {
        for (int i = 0; i < ndofs; ++i)
        {
            const int expected = eq * ndofs + i;
            EXPECT_CLOSE(layout.index(eq, i), expected, 0.0);
        }
    }

    // Specifically check scalar blocks start where we expect
    for (int k = 0; k < num_scalars; ++k)
    {
        const int eq_scalar_k = layout.eq_scalar0 + k;
        for (int i = 0; i < ndofs; ++i)
        {
            const int expected = eq_scalar_k * ndofs + i;
            EXPECT_CLOSE(layout.index(eq_scalar_k, i), expected, 0.0);
        }
    }

    return 0;
}

TEST(DofStateView_ReadsExpectedComponents)
{
    const int dim   = 3;
    const int ndofs = 4;
    const int nscalars = 2;

    Prandtl::StateLayout layout(dim, ndofs, nscalars);
    const int num_eq = dim + 2 + nscalars;             // rho, dim*mom, E, scalars

    std::vector<real_t> U(num_eq * ndofs);

    // Fill equation-blocked storage with a simple pattern:
    //   U(eq, i) = 10*eq + i + 1.0
    for (int eq = 0; eq < num_eq; ++eq)
    {
        for (int i = 0; i < ndofs; ++i)
        {
            U[eq * ndofs + i] = 10.0 * eq + i + 1;
        }
    }

    for (int i = 0; i < ndofs; ++i)
    {
        Prandtl::DofStateView S{U.data(), &layout, i};

        // Mass
        EXPECT_EQ(S.mass(), 10.0 * layout.eq_mass + i + 1);

        // Momentum components
        for (int d = 0; d < dim; ++d)
        {
            const int eq_m = layout.eq_mom[d];
            EXPECT_EQ(S.momentum(d), 10.0 * eq_m + i + 1);
            if (d == 0){
              EXPECT_EQ(S.momentum_x(), S.momentum(d));
            }
            if (d == 1){
              EXPECT_EQ(S.momentum_y(), S.momentum(d));
            }
            if (d == 2){
              EXPECT_EQ(S.momentum_z(), S.momentum(d));
            }
        }

        // Velocity components
        for (int d = 0; d < dim; ++d)
        {
          const int eq_m = layout.eq_mom[d];
          EXPECT_CLOSE(S.velocity(d),(10.0*eq_m+i+1)/(i+1), 1e-14);
          if (d == 0){
            EXPECT_EQ(S.velocity_x(), S.velocity(d));
          }
          if (d == 1){
            EXPECT_EQ(S.velocity_y(), S.velocity(d));
          }
          if (d == 2){
            EXPECT_EQ(S.velocity_z(), S.velocity(d));
          }
        }
        
        // Energy
        EXPECT_CLOSE(S.energy(), 10.0 * layout.eq_energy + i + 1, 1e-14);
        
        // Scalars
        for(int s = 0; s < nscalars; s++){
          const int eq_s = layout.eq_scalar0;
          EXPECT_EQ(S.scalar(s),10.0*(eq_s+s) + i + 1);
        }
    }
    
    return 0;
}

TEST(FieldStateView_ReadWriteRoundTrip)
{
    const int dim        = 3;
    const int ndofs      = 3;
    const int num_scalars = 2;

    Prandtl::StateLayout layout(dim, ndofs, num_scalars);
    const int num_eq = dim + 2 + num_scalars;

    std::vector<real_t> U(num_eq * ndofs, 0.0);
    Prandtl::FieldStateView S{U.data(), &layout};

    // Write using named accessors
    for (int i = 0; i < ndofs; ++i)
    {
        S.mass(i)       = 1.0 + i;
        S.momentum_x(i) = 2.0 + i;
        S.momentum_y(i) = 3.0 + i;
        S.momentum_z(i) = 4.0 + i;
        S.energy(i)     = 5.0 + i;
        S.scalar(0, i)  = 6.0 + i;
        S.scalar(1, i)  = 7.0 + i;
    }

    // Read back
    for (int i = 0; i < ndofs; ++i)
    {
      EXPECT_EQ(S.mass(i),       1.0 + i);
      EXPECT_EQ(S.momentum_x(i), 2.0 + i);
      EXPECT_EQ(S.momentum_y(i), 3.0 + i);
      EXPECT_EQ(S.momentum_z(i), 4.0 + i);
      EXPECT_EQ(S.energy(i),     5.0 + i);
      EXPECT_EQ(S.scalar(0, i),  6.0 + i);
      EXPECT_EQ(S.scalar(1, i),  7.0 + i);
      EXPECT_CLOSE(S.velocity_x(i), (2.0 + i)/(1.0 + i), 1e-16);
      EXPECT_CLOSE(S.velocity_y(i), (3.0 + i)/(1.0 + i), 1e-16);
      EXPECT_CLOSE(S.velocity_z(i), (4.0 + i)/(1.0 + i), 1e-16);
    }

    return 0;
}

