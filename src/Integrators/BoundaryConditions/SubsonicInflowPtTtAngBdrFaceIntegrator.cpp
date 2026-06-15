#include "SubsonicInflowPtTtAngBdrFaceIntegrator.hpp"
#include "BasicOperations.hpp"
#include "Flow.hpp"

namespace Prandtl
{

SubsonicInflowPtTtAngBdrFaceIntegrator::SubsonicInflowPtTtAngBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                               const ActiveGasModel &gasModel_,
                                                                               const NumericalFlux &rsolver, const int Np,
                                                                               const real_t &time,
                                                                               FunctionCoefficient &pt, FunctionCoefficient &Tt,
                                                                               real_t theta, real_t phi, bool t_dependent)
: BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
  pt(pt), Tt(Tt), theta(theta), phi(phi)
{
  V_comps.SetSize(dim);
  
    if (dim > 1)
    {
        V_comps(0) = std::cos(theta * M_PI / 180.0);
        V_comps(1) = std::sin(theta * M_PI / 180.0);
        if (dim > 2)
        {
            V_comps(0) *= std::sin(phi * M_PI / 180.0);
            V_comps(1) *= std::sin(phi * M_PI / 180.0);
            V_comps(3) = std::cos(phi * M_PI / 180.0);
        }
    }
}

SubsonicInflowPtTtAngBdrFaceIntegrator::SubsonicInflowPtTtAngBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                               const ActiveGasModel &gasModel_,
                                                                               const NumericalFlux &rsolver, const int Np,
                                                                               const real_t &time, real_t pt,
                                                                               real_t Tt, real_t theta, real_t phi)
: BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false),
  p0(pt), T0(Tt), theta(theta), phi(phi),
  pt(std::function<real_t(const Vector&)>()), Tt(std::function<real_t(const Vector&)>())
{
    V_comps.SetSize(dim);

    if (dim > 1)
    {
        V_comps(0) = std::cos(theta * M_PI / 180.0);
        V_comps(1) = std::sin(theta * M_PI / 180.0);
        if (dim > 2)
        {
            V_comps(0) *= std::sin(phi * M_PI / 180.0);
            V_comps(1) *= std::sin(phi * M_PI / 180.0);
            V_comps(3) = std::cos(phi * M_PI / 180.0);
        }
    }
}

void SubsonicInflowPtTtAngBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        if (t_dependent)
        {
            pt.SetTime(time);
            Tt.SetTime(time);
        }
        p0 = pt.Eval(Tr, ip);
        T0 = Tt.Eval(Tr, ip);
    }
    Prandtl::PointStateView S1{state1.GetData()};
    Prandtl::PointStateViewRW S2{state2.GetData()};
    auto tot = Prandtl::Flow::isentropic_total_to_static(S1, {p0, T0}, gasModel, thermoTables);
    S2.set_mass(gasModel.L, tot.rho);
    const real_t v = std::sqrt(tot.v2);
    S2.set_momentum(gasModel.L, 0, tot.rho*v*V_comps(0));
    if (dim > 1)
    {
      S2.set_momentum(gasModel.L, 1, tot.rho*v*V_comps(1));
      if (dim > 2)
        S2.set_momentum(gasModel.L, 2, tot.rho*v* V_comps(2));
    }
    S2.set_energy(gasModel.L, tot.energy);
}

void SubsonicInflowPtTtAngBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicInflowPtTtAngBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicInflowPtTtAngBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    fluxFunction.ComputeViscousFlux(state2, dqdx, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
