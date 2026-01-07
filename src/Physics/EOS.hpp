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
    inline real_t gamma(const StateView &S) const
    {
      return phys->gamma;
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

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t entropy(const StateView &S) const
    {
      const real_t p = pressure(S);
      const real_t gamma = phys->gamma;
      // TODO: Augment for correct treatment of passive scalars
      return std::log(p) - gamma * std::log(S.mass());
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const InStateView &S, OutStateView &E) const
    {
      const real_t p = pressure(S);
      const real_t gamma = phys->gamma;
      const real_t rho = S.mass();
      const real_t s = std::log(p) - gamma*std::log(rho);
      const real_t beta = rho / p;
      const real_t v2o2 = kinetic_energy_density(S) / rho;
      const real_t s_rho = (gamma - s)/(gamma - 1) - beta*v2o2;

      E.set_mass(s_rho);
      int dim = S.dim();
      int num_scalars = S.num_scalars();
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(idim, beta * S.velocity(idim));
      }
      E.set_energy(-beta);
      // TODO: Update for correct treatment of passive scalars (depends on ES approach)
      // - Here we should probably set the entropy state to scalar_state / density
      // - If we do that, we need to modify the mass component of the entropy state
      // - Making this fix will make the sensor function sensitive to the scalars
      // - If we need to recover CV from this, lax scalar treatment is a nogo
      for(int iscalar = 0;iscalar < num_scalars;iscalar++){
        E.set_scalar(iscalar, 0.0);
      }
    }

    template<typename InStateView, typename OutStateView>
    inline void grad_entropy_to_grad_prim(const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim) const
    {

      const real_t ke = kinetic_energy_density(S);
      const real_t p = pressure(S);
      const real_t rho = S.mass();
      const real_t rhoE = S.energy();
      const real_t ie = internal_energy_density(S);

      int dim = S.dim();
      int num_scalars = S.num_scalars();

      real_t drho = 0.0;
      for(int idim = 0; idim < dim; idim++){
        dPrim.set_momentum(idim, p/rho * (dE.momentum(idim) + S.velocity(idim)*dE.energy()));
        drho += S.momentum(idim)*dPrim.momentum(idim);
      }
      drho = rho*dE.mass() - dE.energy()*(ke - ie) + rho*drho/p;
      dPrim.set_mass(drho);
      dPrim.set_energy( p/rho * (dPrim.mass() + p*dE.energy()));
      for(int isp = 0; isp < num_scalars; isp++){
        dPrim.set_scalar(isp, 0.0); // just a placeholder for now
      }
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
