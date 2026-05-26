#include "LaxFriedrichsFlux.hpp"

namespace Prandtl
{

  real_t LaxFriedrichsFlux::ComputeVolumeFlux(const mfem::Vector &state1, const mfem::Vector &state2,
                                              const mfem::Vector &metric1, const mfem::Vector &metric2,
                                              const ThermoTablesView &thermoTables,
                                              mfem::Vector &F_tilde)
  {
    return ComputeVolumeFluxKernel(gasModel, thermoTables, state1.GetData(), state2.GetData(),
                                   metric1.GetData(), metric2.GetData(),
                                   F_tilde.GetData());
  }


  real_t LaxFriedrichsFlux::ComputeFaceFlux(const mfem::Vector &state1, const mfem::Vector &state2,
                                            const mfem::Vector &nor, const ThermoTablesView &thermoTables,
                                            mfem::Vector &flux) const
  {
    return ComputeFaceFluxKernel(gasModel, thermoTables, state1.GetData(), state2.GetData(),
                                 nor.GetData(), flux.GetData());
  }
  
}
