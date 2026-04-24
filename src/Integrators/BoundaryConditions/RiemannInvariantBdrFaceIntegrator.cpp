#include "RiemannInvariantBdrFaceIntegrator.hpp"
#include "BasicOperations.hpp"
#include "Flow.hpp"

namespace Prandtl
{
  // Constructor for RiemannInvariantBdrFaceIntegrator with a variable (space- and/or time-dependent) primitive state
  RiemannInvariantBdrFaceIntegrator::RiemannInvariantBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       const ActiveGasModel &gasModel_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time,
                                                                       VectorFunctionCoefficient &prim_state_fun, bool t_dependent)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
    prim_state_fun(prim_state_fun)
  {
    unit_nor.SetSize(dim);
    prim_o.SetSize(num_equations);
    state_o.SetSize(num_equations);
  }
  
  // Constructor for RiemannInvariantBdrFaceIntegrator with a constant primitive state
  RiemannInvariantBdrFaceIntegrator::RiemannInvariantBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       const ActiveGasModel &gasModel_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time, const Vector &prim_state)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false),
    prim_state_fun(num_equations, std::function<void(const Vector&, Vector&)>())
  {
    unit_nor.SetSize(dim);
    prim_o.SetSize(num_equations);
    state_o.SetSize(num_equations);
    prim_o(gasModel.L.eq_mass) = prim_state(gasModel.L.eq_mass);
    prim_o(gasModel.L.eq_energy) = prim_state(gasModel.L.eq_energy);
    size_t voff = gasModel.L.eq_mom[0];
    prim_o(voff) = prim_state(voff);
    if (dim > 1) prim_o(voff+1) = prim_state(voff+1);
    if (dim > 2) prim_o(voff+2) = prim_state(voff+2);
    Prandtl::PointPrimitiveView P{prim_o.GetData()};
    Prandtl::PointStateViewRW S{state_o.GetData()};
    Prandtl::Flow::PrimitiveToConserved(P, S, gasModel);
}

void RiemannInvariantBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    unit_nor = nor;
    Normalize(unit_nor);

    if (!constant)
    {
        if (t_dependent)
            prim_state_fun.SetTime(time);
        
        prim_state_fun.Eval(prim_o, Tr, ip);
        Prandtl::PointPrimitiveView P{prim_o.GetData()};
        Prandtl::PointStateViewRW S{state_o.GetData()};
        Prandtl::Flow::PrimitiveToConserved(P, S, gasModel);
    }
    Prandtl::PointStateView So{state_o.GetData()};
    Prandtl::PointStateView Si{state1.GetData()};
    Prandtl::PointStateViewRW S2{state2.GetData()};
    real_t n[3] = { unit_nor(0), (dim>1 ? unit_nor(1) : 0.0), (dim>2 ? unit_nor(2) : 0.0) };

    Prandtl::Flow::riemann_invariant_outer_state(Si, So, S2, n, gasModel);
}

void RiemannInvariantBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void RiemannInvariantBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void RiemannInvariantBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
