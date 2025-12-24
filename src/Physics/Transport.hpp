#pragma once

#include "mfem.hpp"
#include "Physics.hpp"
#include "GasState.hpp"

namespace Prandtl
{

  // ============================================================================
  // Transport: Simple transport, also support Sutherland model (for now)
  // ============================================================================
  
  struct Transport
  {
    std::shared_ptr<const PhysicsConstants> phys;
    
    MFEM_HOST_DEVICE
    explicit Transport(std::shared_ptr<const PhysicsConstants> pc)
      : phys(std::move(pc))
    { }
    
    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline real_t viscosity(const EOSType &eos, const StateViewType &S) const
    {
#ifdef SUTHERLAND
      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
      const real_t temptr = eos.temperature(S);
      const real_t Trel = temptr / phys->T0;
      const real_t T0pTs = phys->T0 + phys->Ts;
      return phys->mu0 * T0pTs * Trel * std::sqrt(Trel) / (temptr + phys->Ts);
#else
      return phys->mu;
#endif
    }

    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline real_t bulk_viscosity(const EOSType &eos, const StateViewType &S) const
    {
      return phys->mu_bulk;
    }

    // Thermal cond kappa = mu * cp / Pr
    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline real_t thermal_conductivity(const EOSType &eos, const StateViewType &S) const
    {
      return viscosity(eos, S) * eos.cp(S) * phys->PrInverse;
    }
  };
  // TODO: Consider refactoring; would be better (explicit) design
  // struct SutherlandTransport {***} using phys.mu0, phys.T0, phys.Ts, etc.
}
