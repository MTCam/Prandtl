#pragma once

#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{

using namespace mfem;

class SubsonicOutflowPBdrFaceIntegrator : public BdrFaceIntegrator
{
private:
    real_t p;
    real_t V_sq;
    FunctionCoefficient p_fun;
public:
  SubsonicOutflowPBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                    const ActiveGasModel &gasModel_,
                                    const NumericalFlux &rsolver, const int Np, const real_t &time,
                                    FunctionCoefficient &p_fun, bool t_dependent = false);
    SubsonicOutflowPBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                      const ActiveGasModel &gasModel_,
                                      const NumericalFlux &rsolver, const int Np, const real_t &time, real_t p_out);
    
    virtual void ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables) override;

    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables) override;
    virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables) override;
};

}
