#pragma once

#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{

using namespace mfem;

class SubsonicInflowRVBdrFaceIntegrator : public BdrFaceIntegrator
{
private:
    real_t u2;
    real_t r;
    Vector u;
    FunctionCoefficient rho;
    VectorFunctionCoefficient V;
public:
    SubsonicInflowRVBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                      std::shared_ptr<const IdealGasModel> gasModel_,
                                      const NumericalFlux &rsolver, const int Np, const real_t &time,
                                      FunctionCoefficient &rho, VectorFunctionCoefficient &V, bool t_dependent = false);
    SubsonicInflowRVBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                      std::shared_ptr<const IdealGasModel> gasModel_,
                                      const NumericalFlux &rsolver, const int Np, const real_t &time,
                                      real_t rho, const Vector &V);
    
    virtual void ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;

    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
};

}
