#pragma once

#include "BdrFaceIntegrator.hpp"

namespace Prandtl

{

using namespace mfem;

class SymmetryBdrFaceIntegrator : public BdrFaceIntegrator
{
private:
    Vector unit_nor;
    Vector prim;
public:
    SymmetryBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                              const ActiveGasModel &gasModel_,
                              const NumericalFlux &rsolver, int Np, const real_t &time,
                              bool constant = true, bool t_dependent = false);
    virtual real_t ComputeBdrFaceInviscidFlux(const Vector &state1, Vector &state2, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
};


}
