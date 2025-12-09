#pragma once

#include "mfem.hpp"
#include "Physics.hpp"
#include "GasState.hpp"

namespace Prandtl
{

// ============================================================================
// EOS: Ideal single-species gas using PhysicsConstants
// ============================================================================
  struct IdealSingleGasEOS
  {
    PhysicsConstants phys;
    
    // MFEM_HOST_DEVICE
    // IdealSingleGasEOS() = default;
    
    MFEM_HOST_DEVICE
    explicit IdealSingleGasEOS(const PhysicsConstants &pc)
        : phys(pc)
    { }

    // ---- helpers on conservative state --------------------------------------

    MFEM_HOST_DEVICE
    real_t density(const DofStateView &S) const
    {
        return S.mass(); // this is "rho" (mass density)
    }

    MFEM_HOST_DEVICE
    real_t rhoE(const DofStateView &S) const
    {
        // rho*E
        return S.energy();
    }

    MFEM_HOST_DEVICE
    real_t momentum_sq(const DofStateView &S) const
    {
        const int dim = S.L->dim;   // uses state layout
        real_t m2 = 0;
        for (int d = 0; d < dim; ++d)
        {
          const real_t m = S.momentum(d);
          m2 += m * m;
        }
        return m2;
    }

    MFEM_HOST_DEVICE
    real_t kinetic_energy_density(const DofStateView &S) const
    {
        // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
        const real_t rho  = density(S);
        const real_t m2   = momentum_sq(S);
        return 0.5 * m2 / rho;
    }

    MFEM_HOST_DEVICE
    real_t internal_energy_density(const DofStateView &S) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
        return rhoE(S) - kinetic_energy_density(S);
    }

    MFEM_HOST_DEVICE
    real_t specific_internal_energy(const DofStateView &S) const
    {
        // e = (rho*e) / rho
        const real_t rho  = density(S);
        const real_t rhoe = internal_energy_density(S);
        return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    MFEM_HOST_DEVICE
    real_t pressure(const DofStateView &S) const
    {
        // p = (gamma - 1) * (rho*E - 0.5*|rho*u|^2 / rho)
        const real_t rhoe = internal_energy_density(S);
        return phys.gammaM1 * rhoe;
    }

    MFEM_HOST_DEVICE
    real_t temperature(const DofStateView &S) const
    {
        // p = rho*R*T  =>  T = p / (rho*R)
        const real_t rho = density(S);
        const real_t p   = pressure(S);
        return p / (rho * phys.R_gas);
    }

    MFEM_HOST_DEVICE
    real_t sound_speed(const DofStateView &S) const
    {
        // a^2 = gamma * p / rho
        const real_t rho = density(S);
        const real_t p   = pressure(S);
        // device-ready sqrt
        // return mfem::Sqrt(phys.gamma * p / rho);
        return std::sqrt(phys.gamma * p / rho);
    }

    // cp is constant for ideal gas
    MFEM_HOST_DEVICE
    real_t cp(const DofStateView & /*S*/) const
    {
        return phys.cp;
    }

    // just because convenience maybe
    MFEM_HOST_DEVICE
    void velocity(const DofStateView &S, real_t u[3]) const
    {
        const real_t rho = density(S);
        const int dim = S.L->dim;
        for (int d = 0; d < dim; ++d)
        {
            u[d] = S.momentum(d) / rho;
        }
        for (int d = dim; d < 3; ++d)
        {
          u[d] = real_t(0);
        }
    }
  };
}
