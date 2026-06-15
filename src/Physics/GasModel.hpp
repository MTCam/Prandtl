#pragma once

#include "Physics.hpp"
#include "GasState.hpp"
#include "EOS.hpp"
#include "LteEOS.hpp"
#include "Transport.hpp"
#include "LteTransport.hpp"

namespace Prandtl
{

// ============================================================================
// GasModel: Encapsulate EOS/Transport
// ============================================================================
  template <typename EOSImpl, typename TransportImpl>
  struct GasModel
  {

    PhysicsConstants phys;
    StateLayout L;
    EOSImpl eos;
    TransportImpl transport;

    MFEM_HOST_DEVICE GasModel() = default;

    MFEM_HOST_DEVICE
    GasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
             const EOSImpl &eos_in, const TransportImpl &tr_in)
      : phys(phys_in), L(L_in), eos(eos_in), transport(tr_in)
    { };

    MFEM_HOST_DEVICE
    GasModel(const PhysicsConstants &phys_in, const StateLayout &L_in)
      : phys(phys_in), L(L_in)
    { };

    // Utilities and constants etc
    MFEM_HOST_DEVICE
    inline int num_equations() const
    { return L.nequations(); };

    MFEM_HOST_DEVICE
    inline int dim() const
    { return L.dim; };

    // State Access
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t velocity(const StateView &S, int d, const TabStruct &tables) const
    { return S.velocity(L,d);};

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t momentum(const StateView &S, int d, const TabStruct &tables) const
    { return S.momentum(L,d);};

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t density(const StateView &S, const TabStruct &tables) const
    {
      return eos.density(phys, L, S, tables);
    };

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t mass(const StateView &S, const TabStruct &tables) const
    {
      return eos.density(phys, L, S, tables);
    };

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t scalar(const StateView &S, int s, const TabStruct &tables) const
    {
      return S.scalar(L, s);
    };

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t energy(const StateView &S, const TabStruct &tables) const
    {
      return S.energy(L);
    };

    // --- Thermodynamics ------------------------------------------------------
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t pressure(const StateView &S, const TabStruct &tables) const
    {
      return eos.pressure(phys, L, S, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t gamma(const StateView &S, const TabStruct &tables) const
    {
      return eos.gamma(phys, L, S, tables);
    }
 
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t cp(const StateView &S, const TabStruct &tables) const
    {
      return eos.cp(phys, L, S, tables);
    }
    
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t R_gas(const StateView &S, const TabStruct &tables) const
    {
      return eos.R_gas(phys, L, S, tables);
    }
 
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t temperature(const StateView &S, const TabStruct &tables) const
    {
      return eos.temperature(phys, L, S, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const StateView &S, const TabStruct &tables) const
    {
      return eos.sound_speed(phys, L, S, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const StateView &S, const TabStruct &tables) const
    {
      return eos.kinetic_energy_density(phys, L, S, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_from_pressure(const StateView &S, real_t pressure, const TabStruct &tables) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
      return eos.internal_energy_from_pressure(phys, L, S, pressure, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const StateView &S, const TabStruct &tables) const
    {
      return eos.specific_internal_energy(phys, L, S, tables);
    }
    
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const StateView &S,
                                 const real_t *grad_r, const real_t *grad_p,
                                 real_t *grad_t, const TabStruct &tables) const
    {
      return eos.grad_temperature(phys, L, S, grad_r, grad_p, grad_t, tables);
    }
 
    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t entropy(const StateView &S, const TabStruct &tables) const
    {
      return eos.entropy(phys, L, S, tables);
    }

    template<typename InStateView, typename OutStateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline void entropy_state(const InStateView &S, OutStateView &E, const TabStruct &tables) const
    {
      return eos.entropy_state(phys, L, S, E, tables);
    }

    template<typename InStateView, typename OutStateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const InStateView &S, const InStateView &dS,
                                          OutStateView &dPrim, const TabStruct &tables) const
    {
      return eos.grad_entropy_to_grad_prim(phys, L, S, dS, dPrim, tables);
    }

    template<typename InStateView, typename OutStateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const InStateView &Se, OutStateView &Sc, const TabStruct &tables) const
    {
      return eos.entropy_to_conserved(phys, L, Se, Sc, tables);
    }

    template<typename InStateView, typename OutStateView, typename TabStruct>
    inline void primitive_to_conserved(const InStateView &prim, OutStateView &cons, const TabStruct &tables) const
    {
      return eos.primitive_to_conserved(phys, L, prim, cons, tables);
    }

    // --- Transport -----------------------------------------------------------

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t viscosity(const StateView &S, const TabStruct &tables) const
    {
      return transport.viscosity(phys, L, eos, S, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t bulk_viscosity(const StateView &S, const TabStruct &tables) const
    {
      return transport.bulk_viscosity(phys, L, eos, S, tables);
    }

    template<typename StateView, typename TabStruct>
    MFEM_HOST_DEVICE
    inline real_t thermal_conductivity(const StateView &S, const TabStruct &tables) const
    {
      return transport.thermal_conductivity(phys, L, eos, S, tables);
    }
  };

  struct ThermoTablesData
  {
    mfem::Vector lte_table;
    mfem::Vector inv_table;
    mfem::Vector rho_grid;
    mfem::Vector T_grid;
    mfem::Vector e_grid;
  };

  struct ThermoTablesView
  {
    const real_t *lte_table;
    const real_t *inv_table;
    const real_t *rho_grid;
    const real_t *T_grid;
    const real_t *e_grid;
  };

  using IdealGasModel = GasModel<IdealSingleGasEOS, Transport>;
  using LTEGasModel = GasModel<LTEGasEOS, LTETransport>;
#ifdef LTE_EOS
  using ActiveGasModel = LTEGasModel;
#else
  using ActiveGasModel = IdealGasModel;
#endif
  // Bridge helper so old call-sites that only have PhysicsConstants can move over
  // inline IdealGasModel make_ideal_gas_model(std::shared_ptr<const PhysicsConstants> phys)
  // {
  //   return IdealGasModel(phys);
  // }

} // namespace Prandtl
