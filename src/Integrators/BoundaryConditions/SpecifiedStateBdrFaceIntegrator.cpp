#include "SpecifiedStateBdrFaceIntegrator.hpp"

namespace Prandtl
{

// Constructor for SpecifiedStateBdrfaceIntegrator with a variable (space- and/or time-dependent) conservative state
SpecifiedStateBdrFaceIntegrator::SpecifiedStateBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                 const ActiveGasModel &gasModel_,
                                                                 const NumericalFlux &rsolver, const int Np,
                                                                 const real_t &time,
                                                                 VectorFunctionCoefficient &conserv_state_fun_, bool t_dependent)
: BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
  conserv_state_fun(conserv_state_fun_) {}


// Constructor for SpecifiedStateBdrfaceIntegrator with a constant conservative state
SpecifiedStateBdrFaceIntegrator::SpecifiedStateBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                 const ActiveGasModel &gasModel_,
                                                                 const NumericalFlux &rsolver, int Np,
                                                                 const real_t &time, const Vector &conserv_state)
: BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false), 
  const_state(conserv_state), conserv_state_fun(num_equations, std::function<void(const Vector&, Vector&)>()) {}

void SpecifiedStateBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2,
    FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    if (constant)
    {
        state2 = const_state;
    }
    else
    {
        if (t_dependent)
        {
            conserv_state_fun.SetTime(time);
        }
        conserv_state_fun.Eval(state2, Tr, ip);
    }
}

void SpecifiedStateBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SpecifiedStateBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SpecifiedStateBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
