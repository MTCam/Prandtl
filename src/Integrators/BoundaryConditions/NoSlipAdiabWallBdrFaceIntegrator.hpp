#pragma once

#include "SlipWallBdrFaceIntegrator.hpp"

namespace Prandtl
{

using namespace mfem;

class NoSlipAdiabWallBdrFaceIntegrator : public SlipWallBdrFaceIntegrator
{
private:
    real_t qn, v;
    Vector V;
    FunctionCoefficient qn_wall;
    VectorFunctionCoefficient V_wall;
public:
  NoSlipAdiabWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, const IdealGasModel &gasModel_,
                                   const NumericalFlux &rsolver, const int Np, const real_t &time,
                                   FunctionCoefficient &qn_wall, VectorFunctionCoefficient &V_wall,
                                   bool t_dependent = false);
  NoSlipAdiabWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, const IdealGasModel &gasModel_,
                                   const NumericalFlux &rsolver, const int Np, const real_t &time, real_t qn,
                                   const Vector &V);
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx,
                                           const Vector &dqdy, const Vector &dqdz, Vector &fluxN,
                                           const Vector &nor, FaceElementTransformations &Tr,
                                           const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx,
                                           const Vector &dqdy, Vector &fluxN, const Vector &nor,
                                           FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx,
                                           Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr,
                                           const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceLiftingFlux(const Vector &state1, Vector &fluxN, FaceElementTransformations &Tr,
                                           const IntegrationPoint &ip) override;
};

}
