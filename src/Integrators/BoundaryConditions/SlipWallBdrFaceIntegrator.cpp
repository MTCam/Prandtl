#include "SlipWallBdrFaceIntegrator.hpp"
#include "BasicOperations.hpp"
#include "Flow.hpp"

namespace Prandtl
{

  SlipWallBdrFaceIntegrator::SlipWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                       const ActiveGasModel &gasModel_,
                                                       const NumericalFlux &rsolver, int Np, const real_t &time,
                                                       bool constant, bool t_dependent)
    : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, constant, t_dependent)
{
    unit_nor.SetSize(rsolver.GetFluxFunction().dim);
    prim.SetSize(rsolver.GetFluxFunction().num_equations);
}


real_t SlipWallBdrFaceIntegrator::ComputeBdrFaceInviscidFlux(const Vector &state1, Vector &state2,
    Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
  return Prandtl::BC::SlipWallInviscidFluxKernel(gasModel, state1.HostRead(),
                                                 nor.HostRead(), fluxN.HostWrite());
}

}
