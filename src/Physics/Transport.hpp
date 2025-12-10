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
    PhysicsConstants phys;
    
    // MFEM_HOST_DEVICE
    // Transport() = default;
    
    MFEM_HOST_DEVICE
    explicit Transport(const PhysicsConstants &pc)
      : phys(pc)
    { }
    
    // dynamic visc mu: constant (or Sutherland)
    MFEM_HOST_DEVICE
    real_t viscosity(real_t temperature) const
    {
#ifdef SUTHERLAND
      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
      real_t Trel = temperature / phys.T0;
      real_t T0pTs = phys.T0 + phys.Ts;
      // return phys.mu0 * T0pTs * Trel * mfem::Sqrt(Trel) / (temperature + phys.Ts);
      return phys.mu0 * T0pTs * Trel * std::sqrt(Trel) / (temperature + phys.Ts);
#else
      return phys.mu;
#endif
    }
    
    // Thermal cond kappa = mu * cp / Pr
    MFEM_HOST_DEVICE
    real_t thermal_conductivity(real_t temperature, real_t cp) const
    {
      return viscosity(temperature) * cp * phys.PrInverse;
    }
    
  };
  // Hrm, probably not strictly needed, but would be better (explicit) design
  // struct SutherlandTransport {***} using phys.mu0, phys.T0, phys.Ts, etc.
}
