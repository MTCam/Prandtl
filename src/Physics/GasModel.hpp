#pragma once

#include "mfem.hpp"
#include "Physics.hpp"
#include "GasState.hpp"
#include "EOS.hpp"
#include "Transport.hpp"

namespace Prandtl
{

// ============================================================================
// GasModel: Encapsulate EOS/Transport
// ============================================================================

  template <typename EOSImpl, typename TransportImpl>
  struct GasModel
  {
    EOSImpl eos;
    TransportImpl transport;
    
    MFEM_HOST_DEVICE
    GasModel() = default;

    MFEM_HOST_DEVICE
    GasModel(const EOSImpl &eos_in, const TransportImpl &tr_in)
        : eos(eos_in), transport(tr_in)
    { }

    MFEM_HOST_DEVICE
    explicit GasModel(std::shared_ptr<const PhysicsConstants> phys)
        : eos(EOSImpl(phys)), transport(TransportImpl(phys))
    { }

    // --- Thermodynamics ------------------------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t pressure(const StateView &S) const
    {
        return eos.pressure(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t gamma(const StateView &S) const
    {
        return eos.gamma(S);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t temperature(const StateView &S) const
    {
        return eos.temperature(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const StateView &S) const
    {
        return eos.sound_speed(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t density(const StateView &S) const
    {
        return eos.density(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const StateView &S) const
    {
      return eos.kinetic_energy_density(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const StateView &S) const
    {
        return eos.specific_internal_energy(S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const int ndim, const StateView &S,
                                 const real_t *grad_r, const real_t *grad_p,
                                 real_t *grad_t) const
    {
      return eos.grad_temperature(ndim, S, grad_r, grad_p, grad_t);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t entropy(const StateView &S)
    {
      return eos.entropy(S);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const InStateView &S, OutStateView &E) const
    {
      return eos.entropy_state(S, E);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const InStateView &S, const InStateView &dS,
                                          OutStateView &dPrim) const
    {
      return eos.grad_entropy_to_grad_prim(S, dS, dPrim);
    }

    // --- Transport -----------------------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t viscosity(const StateView &S) const
    {
      return transport.viscosity(eos, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t bulk_viscosity(const StateView &S) const
    {
      return transport.bulk_viscosity(eos, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t thermal_conductivity(const StateView &S) const
    {
      return transport.thermal_conductivity(eos, S);
    }
  };

  // Current concrete choice: ideal single-species gas + constant transport
  using IdealGasModel = GasModel<IdealSingleGasEOS, Transport>;
  
  // Bridge helper so old call-sites that only have PhysicsConstants can move over
  inline IdealGasModel make_ideal_gas_model(std::shared_ptr<const PhysicsConstants> phys)
  {
    return IdealGasModel(phys);
  }

} // namespace Prandtl
