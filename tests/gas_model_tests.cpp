#include "unit_test.hpp"

#include "GasModel.hpp"

#include <vector>
#include <cmath>

using namespace Prandtl;

// Helper to fill a single-DOF conservative state in `U`.
//
// Layout (equation-blocked):
//   eq_mass   -> rho
//   eq_mom[d] -> rho * u_d
//   eq_energy -> rhoE = e_int_density + 0.5 * rho * |u|^2
static void
fill_single_dof_state(StateLayout &layout,
                      std::vector<real_t> &U,
                      int dim,
                      real_t rho,
                      const real_t u[3],
                      real_t e_int_density)
{
    const int ndofs  = 1;
    const int num_eq = layout.eq_energy + 1; // dim+2 when no scalars
    (void)ndofs; // silence unused warning, kept for clarity

    // Zero everything for safety.
    for (int eq = 0; eq < num_eq; ++eq)
    {
        U[layout.index(eq, 0)] = real_t(0);
    }

    // Mass
    U[layout.index(layout.eq_mass, 0)] = rho;

    // Momentum: rho * u_d
    for (int d = 0; d < dim; ++d)
    {
        U[layout.index(layout.eq_mom[d], 0)] = rho * u[d];
    }

    // Kinetic energy density: 0.5 * rho * |u|^2
    real_t vsq = 0;
    for (int d = 0; d < dim; ++d)
    {
        vsq += u[d] * u[d];
    }
    const real_t kinetic = real_t(0.5) * rho * vsq;

    // Total energy density: rhoE = e_int_density + kinetic
    U[layout.index(layout.eq_energy, 0)] = e_int_density + kinetic;
}

// -----------------------------------------------------------------------------
// GasModel thermo test: pressure, temperature, sound-speed, density, and
// specific internal energy for an ideal single-species gas in dim = 1,2,3.
//
// For each dimension:
//  - Choose rho, u, and internal energy density e_int such that
//        e_int = 1 / (gamma - 1)
//
//    so that, for the ideal gas, the pressure should be
//        p = (gamma - 1) * e_int = 1
//
//  - Check:
//       * gas.pressure(S) == 1
//       * gas.temperature(S) == p / (rho * R_gas)
//       * gas.sound_speed(S) == sqrt(gamma * p / rho)
//       * gas.density(S) == rho
//       * gas.specific_internal_energy(S) == e_int / rho
// -----------------------------------------------------------------------------
TEST(GasModel_IdealGas_EOS)
{
    // Reasonably physical values
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    PhysicsConstants phys(gamma, Pr, R_gas, mu);
    IdealGasModel gas(phys);

    const real_t tol = 1.0e-12;

    for (int dim = 1; dim <= 3; ++dim)
    {
        const int ndofs = 1;
        StateLayout layout(dim, ndofs);  // no scalars
        const int num_eq = layout.eq_energy + 1; // dim+2

        std::vector<real_t> U(num_eq * ndofs);

        // Fixed density.
        const real_t rho = 2.0;

        // Velocity (only first 'dim' components matter).
        const real_t u[3] = {10.0, -3.0, 5.0};

        // Internal energy density chosen so that p = 1, regardless of u.
        //
        // For ideal gas with total energy density:
        //   rhoE = e_int + 0.5 * rho * |u|^2
        //
        // p = (gamma - 1) * (rhoE - 0.5 * rho * |u|^2)
        //   = (gamma - 1) * e_int
        //
        // So choose e_int = 1 / (gamma - 1) => p = 1.
        const real_t e_int_density = 1.0 / (gamma - 1.0);

        fill_single_dof_state(layout, U, dim, rho, u, e_int_density);

        DofStateView S(U.data(), &layout, 0);

        // Pressure
        const real_t p = gas.pressure(S);
        EXPECT_CLOSE(p, 1.0, tol);

        // Temperature: p = rho * R * T => T = p / (rho * R)
        const real_t T_expected = p / (rho * R_gas);
        const real_t T = gas.temperature(S);
        EXPECT_CLOSE(T, T_expected, tol);

        // Sound speed: a^2 = gamma * p / rho
        const real_t a_expected = std::sqrt(gamma * p / rho);
        const real_t a = gas.sound_speed(S);
        EXPECT_CLOSE(a, a_expected, tol);

        // Density should just be rho
        const real_t rho_out = gas.density(S);
        EXPECT_CLOSE(rho_out, rho, tol);

        // Specific internal energy e = e_int_density / rho
        const real_t e_expected = e_int_density / rho;
        const real_t e_si = gas.specific_internal_energy(S);
        EXPECT_CLOSE(e_si, e_expected, tol);
    }

    return 0;
}

// -----------------------------------------------------------------------------
// GasModel transport test: verify that GasModel's viscosity and thermal
// conductivity are consistent with EOS + Transport.
//
// For a given state S:
//   - Let T  = eos.temperature(S)
//   - Let cp = eos.cp(S)
//   - Let mu_expected = transport.viscosity(T)
//   - Let k_expected  = transport.thermal_conductivity(T, cp)
//
// We expect:
//   - gas.viscosity(S) == mu_expected
//   - gas.thermal_conductivity(S) == k_expected
//
// This works for both constant and Sutherland transport, because it uses
// the same Transport implementation as GasModel internally.
// -----------------------------------------------------------------------------
TEST(GasModel_IdealGas_Transport)
{
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    PhysicsConstants phys(gamma, Pr, R_gas, mu);

    IdealGasModel      gas(phys);
    IdealSingleGasEOS  eos(phys);
    Transport          transport(phys);

    const real_t tol = 1.0e-12;

    // Just pick dim = 3 here; the exact dimension doesn't really matter
    // as long as the state is consistent.
    const int dim   = 3;
    const int ndofs = 1;
    StateLayout layout(dim, ndofs);

    const int num_eq = layout.eq_energy + 1;
    std::vector<real_t> U(num_eq * ndofs);

    const real_t rho = 1.5;
    const real_t u[3] = {50.0, -20.0, 5.0};

    // Again choose e_int_density such that p = 1 (just for convenience):
    const real_t e_int_density = 1.0 / (gamma - 1.0);

    fill_single_dof_state(layout, U, dim, rho, u, e_int_density);

    DofStateView S(U.data(), &layout, 0);

    // Use the EOS + Transport directly to compute the expected results
    const real_t T  = eos.temperature(S);
    const real_t cp = eos.cp(S);

    const real_t mu_expected = transport.viscosity(T);
    const real_t k_expected  = transport.thermal_conductivity(T, cp);

    const real_t mu_gas = gas.viscosity(S);
    const real_t k_gas  = gas.thermal_conductivity(S);

    EXPECT_CLOSE(mu_gas, mu_expected, tol);
    EXPECT_CLOSE(k_gas,  k_expected,  tol);

    return 0;
}
