# pragma once

#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{
using namespace mfem;

class SupersonicOutflowBdrFaceIntegrator : public BdrFaceIntegrator
{
public:
    SupersonicOutflowBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                       std::shared_ptr<const IdealGasModel> gasModel_,
                                       std::shared_ptr<const StateLayout> stateLayout_,
                                       const NumericalFlux &rsolver, const int Np, const real_t &time);
    virtual void ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override; 
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override; 
};

}
