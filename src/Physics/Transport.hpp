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
    
    // dynamic visc mu: constant (or Sutherland)
    //MFEM_HOST_DEVICE
    //inline real_t viscosity(real_t temperature) const
    //{
    //#ifdef SUTHERLAND
    //      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
    //      real_t Trel = temperature / phys.T0;
    //      real_t T0pTs = phys.T0 + phys.Ts;
    //      return phys.mu0 * T0pTs * Trel * std::sqrt(Trel) / (temperature + phys.Ts);
    //#else
    //      return phys.mu;
    //#endif
    //    }

    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline real_t viscosity(const EOSType &eos, const StateViewType &S) const
    {
#ifdef SUTHERLAND
      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
      real_t Trel = eos.temperature(S) / phys->T0;
      real_t T0pTs = phys->T0 + phys->Ts;
      return phys->mu0 * T0pTs * Trel * std::sqrt(Trel) / (temperature + phys->Ts);
#else
      return phys->mu;
#endif
    }
    
    // dynamic visc mu: constant (or Sutherland)
    // MFEM_HOST_DEVICE
    // inline real_t bulk_viscosity(real_t temperature) const
    // {
    //   return phys.mu_bulk;
    // }

    // dynamic visc mu: constant (or Sutherland)
    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline real_t bulk_viscosity(const EOSType &eos, const StateViewType &S) const
    {
      return phys->mu_bulk;
    }

    // Thermal cond kappa = mu * cp / Pr
    // MFEM_HOST_DEVICE
    // inline real_t thermal_conductivity(real_t temperature, real_t cp) const
    // {
    //  return viscosity(temperature) * cp * phys.PrInverse;
    // }
    
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
