#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"

namespace Prandtl
{

  // ============================================================================
  // Transport: Simple transport, also support Sutherland model (for now)
  // ============================================================================
  
  struct Transport
  {
    
    template<typename EOSType, typename StateViewType, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t viscosity(const PhysicsConstants &phys, const StateLayout &L,
                            const EOSType &eos, const StateViewType &S, const TabStruct &tables) const
    {
#ifdef SUTHERLAND
      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
      const real_t temptr = eos.temperature(phys, L, S, tables);
      const real_t Trel = temptr / phys.T0;
      const real_t T0pTs = phys.T0 + phys.Ts;
      return phys.mu0 * T0pTs * Trel * std::sqrt(Trel) / (temptr + phys.Ts);
#else
      return phys.mu;
#endif
    }

    template<typename EOSType, typename StateViewType, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t bulk_viscosity(const PhysicsConstants &phys, const StateLayout &L,
                                 const EOSType &eos, const StateViewType &S, const TabStruct &tables) const
    {
      return phys.mu_bulk;
    }

    // Thermal cond kappa = mu * cp / Pr
    template<typename EOSType, typename StateViewType, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t thermal_conductivity(const PhysicsConstants &phys, const StateLayout &L,
                                       const EOSType &eos, const StateViewType &S, const TabStruct &tables) const
    {
      return viscosity(phys, L, eos, S, tables) * eos.cp(phys, L, S, tables) * phys.PrInverse;
    }
  };
  // TODO: Consider refactoring; would be better (explicit) design
  // struct SutherlandTransport {***} using phys.mu0, phys.T0, phys.Ts, etc.
}
