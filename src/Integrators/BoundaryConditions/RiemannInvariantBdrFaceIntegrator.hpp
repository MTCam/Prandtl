#pragma once

#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{

using namespace mfem;

class RiemannInvariantBdrFaceIntegrator : public BdrFaceIntegrator
{
private:
  VectorFunctionCoefficient prim_state_fun;
  Vector prim_o;
  Vector state_o;
  Vector unit_nor;

public:
    RiemannInvariantBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                      std::shared_ptr<const IdealGasModel> gasModel_,
                                      std::shared_ptr<const StateLayout> stateLayout_,
                                      const NumericalFlux &rsolver, const int Np, const real_t &time,
                                      VectorFunctionCoefficient &prim_state_fun, bool t_dependent = false);
    RiemannInvariantBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                      std::shared_ptr<const IdealGasModel> gasModel_,
                                      std::shared_ptr<const StateLayout> stateLayout_,
                                      const NumericalFlux &rsolver, const int Np, const real_t &time, const Vector &prim_state);
    virtual void ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
};

}
