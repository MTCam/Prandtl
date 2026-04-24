#include "SupersonicOutflowBdrFaceIntegrator.hpp"

namespace Prandtl
{

  SupersonicOutflowBdrFaceIntegrator::SupersonicOutflowBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                         const ActiveGasModel &gasModel_,
                                                                         const NumericalFlux &rsolver, const int Np,
                                                                         const real_t &time)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false) {}

void SupersonicOutflowBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    state2 = state1;
}

void SupersonicOutflowBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SupersonicOutflowBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SupersonicOutflowBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
