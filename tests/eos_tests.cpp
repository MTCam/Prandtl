#include "unit_test.hpp"
#include "test_helpers.hpp"

#include "Physics.hpp"
#include "GasState.hpp"
#include "EOS.hpp"

#include <vector>
#include <cmath>

using namespace Prandtl;

// -----------------------------------------------------------------------------
// EOS test: pressure, temperature, and sound-speed for an ideal single gas.
//
// For each dimension dim=1,2,3:
//
//  - Choose rho, u, and internal energy density e_int such that
//        e_int = 1 / (gamma - 1)
//
//    so that, for the ideal gas, the pressure should be
//        p = (gamma - 1) * e_int = 1
//
//    regardless of the kinetic energy contribution from velocity.
//
//  - Check:
//       * eos.pressure(S) == 1
//       * eos.temperature(S) == p / (rho * R_gas)
//       * eos.sound_speed(S) == sqrt(gamma * p / rho)
//  - Then change velocity (keeping e_int fixed) and confirm that p, T, a
//    are unchanged.
// -----------------------------------------------------------------------------
TEST(IdealGas_EOS)
{
    // Reasonably physical values, but the test only relies on consistency.
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    IdealSingleGasEOS eos{phys};

    const real_t tol = 1.0e-12;

    for (int dim = 1; dim <= 3; ++dim)
    {
        const int ndofs = 1;
        StateLayout layout(dim, ndofs);  // no scalars
        const int num_eq = layout.eq_energy + 1; // dim+2

        std::vector<real_t> U(num_eq * ndofs);

        // Fixed density.
        const real_t rho = 2.0;

        // First velocity set (only first 'dim' components matter).
        const real_t u1[3] = {10.0, -3.0, 5.0};

        // Internal energy *density* chosen so that p = 1, regardless of u.
        //
        // For an ideal gas, with total energy density:
        //   rhoE = e_int + 0.5 * rho * |u|^2
        // we have:
        //   p = (gamma - 1) * (rhoE - 0.5 * rho * |u|^2)
        //     = (gamma - 1) * e_int
        //
        // So choosing e_int = 1 / (gamma - 1) implies p = 1.
        const real_t e_int_density = 1.0 / (gamma - 1.0);

        // --- Case 1: velocity = u1 -----------------------------------------
        fill_single_dof_state(layout, U, dim, rho, u1, e_int_density);

        DofStateView S1(U.data(), &layout, 0);

        const real_t p1 = eos.pressure(S1);
        EXPECT_CLOSE(p1, 1.0, tol);

        const real_t T1_expected = p1 / (rho * R_gas);
        const real_t T1 = eos.temperature(S1);
        EXPECT_CLOSE(T1, T1_expected, tol);

        const real_t a1_expected = std::sqrt(gamma * p1 / rho);
        const real_t a1 = eos.sound_speed(S1);
        EXPECT_CLOSE(a1, a1_expected, tol);

        // --- Case 2: different velocity, same e_int_density ----------------
        const real_t u2[3] = {-4.0, 7.0, 1.0};

        fill_single_dof_state(layout, U, dim, rho, u2, e_int_density);

        DofStateView S2(U.data(), &layout, 0);

        const real_t p2 = eos.pressure(S2);
        EXPECT_CLOSE(p2, 1.0, tol);  // should be unchanged

        const real_t T2_expected = p2 / (rho * R_gas);
        const real_t T2 = eos.temperature(S2);
        EXPECT_CLOSE(T2, T2_expected, tol);

        const real_t a2_expected = std::sqrt(gamma * p2 / rho);
        const real_t a2 = eos.sound_speed(S2);
        EXPECT_CLOSE(a2, a2_expected, tol);
    }

    return 0;
}

// -----------------------------------------------------------------------------
// EOS test: grad_temperature for an ideal single gas.
//
// For ideal gas: T = p / (rho * R)
// => grad(T) = (1/R) * ( grad(p)/rho - p * grad(rho) / rho^2 )
//
// We test two analytic cases in dim=1,2,3:
//   A) grad(rho)=0, grad(p)!=0
//   B) grad(p)=0, grad(rho)!=0
// -----------------------------------------------------------------------------
TEST(IdealGas_EOS_GradTemperature)
{
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    IdealSingleGasEOS eos{phys};

    const real_t tol = 1.0e-12;

    // Choose a consistent thermodynamic state (pressure set via e_int_density).
    const real_t rho = 2.0;
    const real_t p   = 5.0;

    // Choose internal energy density so EOS returns exactly p independent of u:
    //   p = (gamma-1) * e_int_density  =>  e_int_density = p/(gamma-1)
    const real_t e_int_density = p / (gamma - 1.0);

    // Velocity is irrelevant to T when p and rho are set, but keep nonzero u
    // so this test doesn't accidentally rely on a special-case u=0 path.
    const real_t u[3] = {1.0, -2.0, 0.5};

    for (int dim = 1; dim <= 3; ++dim)
    {
        const int ndofs = 1;
        StateLayout layout(dim, ndofs);  // no scalars
        const int num_eq = layout.eq_energy + 1; // dim+2
        real_t ke = u[0]*u[0];
        for(int idim = 1; idim < dim; idim++){
          ke += u[idim]*u[idim];
        }
        ke *= 0.5*rho;
        std::vector<real_t> U(num_eq * ndofs);
        fill_single_dof_state(layout, U, dim, rho, u, e_int_density);

        DofStateView S(U.data(), &layout, 0);

        // Sanity: ensure EOS returns the intended p, so T is consistent.
        const real_t p_check = eos.pressure(S);
        EXPECT_CLOSE(p_check, p, tol);

        // --- Case A: grad(rho)=0, grad(p)!=0 -------------------------------
        {
            real_t grad_rho[3] = {0.0, 0.0, 0.0};
            real_t grad_p[3]   = {1.0, 2.0, 3.0};

            // Initialize with sentinels so we detect if EOS overwrites.
            real_t grad_t[3]   = {99.0, 99.0, 99.0};

            eos.grad_temperature(dim, S, grad_rho, grad_p, grad_t);

            for (int d = 0; d < dim; ++d)
            {
                const real_t expected = (1.0 / R_gas) * (grad_p[d] / rho);
                EXPECT_CLOSE(grad_t[d], expected, tol);
            }
            for (int d = dim; d < 3; ++d)
            {
                EXPECT_CLOSE(grad_t[d], grad_t[d], tol);
            }
        }

        // --- Case B: grad(p)=0, grad(rho)!=0 -------------------------------
        {
            real_t grad_rho[3] = {1.0, 2.0, 3.0};
            real_t grad_p[3]   = {0.0, 0.0, 0.0};

            real_t grad_t[3]   = {99.0, 99.0, 99.0};

            eos.grad_temperature(dim, S, grad_rho, grad_p, grad_t);

            for (int d = 0; d < dim; ++d)
            {
                const real_t expected =
                    (1.0 / R_gas) * (-(p / (rho * rho)) * grad_rho[d]);
                EXPECT_CLOSE(grad_t[d], expected, tol);
            }
            for (int d = dim; d < 3; ++d)
            {
                EXPECT_CLOSE(grad_t[d], grad_t[d], tol);
            }
        }
    }

    return 0;
}

// -----------------------------------------------------------------------------
// EOS helper-function coverage: momentum_sq, kinetic/internal energy, cp, velocity().
// -----------------------------------------------------------------------------
TEST(IdealGas_EOS_HelperFunctions)
{
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    std::shared_ptr<PhysicsConstants> phys =
      std::make_shared<PhysicsConstants>(gamma, Pr, R_gas, mu);

    IdealSingleGasEOS eos{phys};

    const real_t tol = 1.0e-12;

    // Pick nontrivial state parameters to avoid accidental “special-case” passes.
    const real_t rho = 1.5;
    const real_t u[3] = { 1.0, -2.0, 0.5 };   // truncated by dim
    const real_t e_int_density = 7.0;         // internal energy density (rho * e)

    for (int dim = 1; dim <= 3; ++dim)
    {
        const int ndofs = 1;
        StateLayout layout(dim, ndofs);  // no scalars
        const int num_eq = layout.eq_energy + 1; // dim + 2

        std::vector<real_t> U(num_eq * ndofs);
        fill_single_dof_state(layout, U, dim, rho, u, e_int_density);

        DofStateView S(U.data(), &layout, 0);

        // Compute expected values analytically.
        real_t u2 = 0.0;
        real_t mom_sq = 0.0;
        for (int d = 0; d < dim; ++d)
        {
            u2 += u[d] * u[d];
            const real_t mom_d = rho * u[d];
            mom_sq += mom_d * mom_d;
        }

        const real_t ke_expected = 0.5 * rho * u2;
        const real_t ie_expected = e_int_density;
        const real_t sie_expected = ie_expected / rho;
        const real_t cp_expected = gamma * R_gas / (gamma - 1.0);

        // ---- EOS helper checks ------------------------------------------------

        EXPECT_CLOSE(eos.momentum_sq(S), mom_sq, tol);
        EXPECT_CLOSE(eos.kinetic_energy_density(S), ke_expected, tol);
        EXPECT_CLOSE(eos.internal_energy_density(S), ie_expected, tol);
        EXPECT_CLOSE(eos.specific_internal_energy(S), sie_expected, tol);
        EXPECT_CLOSE(eos.cp(S), cp_expected, tol);

        // EOS velocity helper
        real_t u_out[3] = { 99.0, 99.0, 99.0 };
        eos.velocity(S, u_out);
        for (int d = 0; d < dim; ++d)
        {
            EXPECT_CLOSE(u_out[d], u[d], tol);
        }
    }

    return 0;
}
