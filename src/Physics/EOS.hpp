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
    std::shared_ptr<const PhysicsConstants> phys;
    
    MFEM_HOST_DEVICE
    explicit IdealSingleGasEOS(std::shared_ptr<const PhysicsConstants> pc)
      : phys(std::move(pc))
    { }

    // ---- helpers on conservative state --------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t density(const StateView &S) const
    {
        return S.mass(); // this is "rho" (mass density)
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t rhoE(const StateView &S) const
    {
        // rho*E
        return S.energy();
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t momentum_sq(const StateView &S) const
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

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const StateView &S) const
    {
        // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
        const real_t rho  = density(S);
        const real_t m2   = momentum_sq(S);
        return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_density(const StateView &S) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
        return rhoE(S) - kinetic_energy_density(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const StateView &S) const
    {
        // e = (rho*e) / rho
        const real_t rho  = density(S);
        const real_t rhoe = internal_energy_density(S);
        return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t pressure(const StateView &S) const
    {
        // p = (gamma - 1) * (rho*E - 0.5*|rho*u|^2 / rho)
        const real_t rhoe = internal_energy_density(S);
        return phys->gammaM1 * rhoe;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t temperature(const StateView &S) const
    {
        // p = rho*R*T  =>  T = p / (rho*R)
        const real_t rho = density(S);
        const real_t p   = pressure(S);
        return p / (rho * phys->R_gas);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const int dim, const StateView &S, const real_t *grad_rho,
                          const real_t *grad_p, real_t *grad_t) const
    {
      const real_t rho = density(S);
      const real_t pressor = pressure(S)/rho;
      const real_t cv = cp(S)/phys->gamma;
      const real_t fac = phys->gammaM1Inverse/(cv*rho);
      for(int i = 0; i < dim; i++){
        grad_t[i] = fac*(grad_p[i] - pressor*grad_rho[i]);
      }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const StateView &S) const
    {
        // a^2 = gamma * p / rho
        const real_t rho = density(S);
        const real_t p   = pressure(S);
        return std::sqrt(phys->gamma * p / rho);
    }

    // cp is constant for ideal gas
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t cp(const StateView & /*S*/) const
    {
        return phys->cp;
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const StateView &S, real_t u[3]) const
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
