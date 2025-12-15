#include "unit_test.hpp"
#include "Physics.hpp"
#include "Transport.hpp"

#include <cmath>

using Prandtl::real_t;
using Prandtl::PhysicsConstants;
using Prandtl::Transport;

TEST(Transport_mu_kappa)
{
    // Reasonably physical values (exact numbers not crucial for the test)
    const real_t gamma = 1.4;
    const real_t Pr    = 0.72;
    const real_t R_gas = 287.0;
    const real_t mu    = 1.8e-5;

    PhysicsConstants phys(gamma, Pr, R_gas, mu);
    Transport transport(phys);

    // Two distinct temperatures to probe temperature dependence
    const real_t T1 = phys.T0;
    const real_t T2 = 2.0 * phys.T0;

    const real_t cp  = phys.cp;
    const real_t tol = 1.0e-12;

#ifdef SUTHERLAND
    // -------------------------------------------------------------------------
    // Sutherland-law behavior
    // -------------------------------------------------------------------------
    auto sutherland_mu = [&](real_t T) -> real_t {
        const real_t Trel  = T / phys.T0;
        const real_t T0pTs = phys.T0 + phys.Ts;
        // Match the implementation in Transport::viscosity
        return phys.mu0 * T0pTs * Trel * std::sqrt(Trel) / (T + phys.Ts);
    };

    const real_t mu1_expected = sutherland_mu(T1);
    const real_t mu2_expected = sutherland_mu(T2);

    const real_t mu1 = transport.viscosity(T1);
    const real_t mu2 = transport.viscosity(T2);

    // Check viscosity matches Sutherland formula
    EXPECT_CLOSE(mu1, mu1_expected, tol);
    EXPECT_CLOSE(mu2, mu2_expected, tol);

    // In Sutherland mode, viscosity should *not* be constant in T
    EXPECT_TRUE(std::abs(mu1 - mu2) > 1.0e-16);

    // Thermal conductivity: k = mu(T) * cp / Pr
    const real_t k1_expected = mu1_expected * cp * phys.PrInverse;
    const real_t k2_expected = mu2_expected * cp * phys.PrInverse;

    const real_t k1 = transport.thermal_conductivity(T1, cp);
    const real_t k2 = transport.thermal_conductivity(T2, cp);

    EXPECT_CLOSE(k1, k1_expected, tol);
    EXPECT_CLOSE(k2, k2_expected, tol);

#else
    // -------------------------------------------------------------------------
    // Constant-transport behavior
    // -------------------------------------------------------------------------
    const real_t mu1 = transport.viscosity(T1);
    const real_t mu2 = transport.viscosity(T2);

    // Viscosity should be equal to phys.mu for any T
    EXPECT_CLOSE(mu1, phys.mu, tol);
    EXPECT_CLOSE(mu2, phys.mu, tol);
    EXPECT_CLOSE(mu1, mu2,    tol);

    // Thermal conductivity: constant k = mu * cp / Pr
    const real_t k_expected = phys.mu * cp * phys.PrInverse;

    const real_t k1 = transport.thermal_conductivity(T1, cp);
    const real_t k2 = transport.thermal_conductivity(T2, cp);

    EXPECT_CLOSE(k1, k_expected, tol);
    EXPECT_CLOSE(k2, k_expected, tol);
#endif

    return 0;
}
