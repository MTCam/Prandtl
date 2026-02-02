#include "SubsonicOutflowPBdrFaceIntegrator.hpp"

namespace Prandtl
{

SubsonicOutflowPBdrFaceIntegrator::SubsonicOutflowPBdrFaceIntegrator(
    std::shared_ptr<LiftingScheme> liftingScheme, const NumericalFlux &rsolver, const int Np, const real_t &time, real_t gamma, FunctionCoefficient &p_fun, bool t_dependent)
    : BdrFaceIntegrator(liftingScheme, rsolver, Np, time, gamma, false, t_dependent),
    p_fun(p_fun) {}

SubsonicOutflowPBdrFaceIntegrator::SubsonicOutflowPBdrFaceIntegrator(
    std::shared_ptr<LiftingScheme> liftingScheme, const NumericalFlux &rsolver, const int Np, const real_t &time, real_t gamma, real_t p)
    : BdrFaceIntegrator(liftingScheme, rsolver, Np, time, gamma, true, false),
    p(p), p_fun(std::function<real_t(const Vector&)>()) {}


void SubsonicOutflowPBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    if (!constant)
    {
        if (t_dependent)
        {
            p_fun.SetTime(time);
        }
        p = p_fun.Eval(Tr, ip);
    }
    Vector vel(dim);
    for(int i = 0;i < dim;i++) vel(i) = state1(i+1) / state1(0);
    state2 = state1;
    state2(num_equations - 1) = p * gammaM1Inverse + 0.5 * state1(0) * (vel * vel);
}

void SubsonicOutflowPBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicOutflowPBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicOutflowPBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
