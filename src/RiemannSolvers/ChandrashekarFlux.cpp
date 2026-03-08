#include "ChandrashekarFlux.hpp"
#include "BasicOperations.hpp"


namespace Prandtl
{

  ChandrashekarFlux::ChandrashekarFlux(const NavierStokesFlux &fluxFunction, const IdealGasModel &gasModel_)
    : NumericalFlux(fluxFunction), gasModel(gasModel_)
  {
    metric.SetSize(dim);
  }
  
  real_t ChandrashekarFlux::ComputeVolumeFlux(const Vector &state1, const Vector &state2,
                                              const Vector &metric1, const Vector &metric2,
                                              Vector &F_tilde)
  {
    return ComputeVolumeFluxKernel(gasModel, state1.GetData(), state2.GetData(),
                                   metric1.GetData(), metric2.GetData(),
                                   F_tilde.GetData());
  }


  real_t ChandrashekarFlux::ComputeFaceFlux(const Vector &state1, const Vector &state2,
                                            const Vector &nor, Vector &flux) const
  {
    return ComputeFaceFluxKernel(gasModel, state1.GetData(), state2.GetData(),
                                 nor.GetData(), flux.GetData());
  }

}
