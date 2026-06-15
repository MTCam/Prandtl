#include "SubsonicOutflowPBdrFaceIntegrator.hpp"

namespace Prandtl
{

  SubsonicOutflowPBdrFaceIntegrator::SubsonicOutflowPBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       const ActiveGasModel &gasModel_,
                                                                       const NumericalFlux &rsolver, const int Np, const real_t &time,
                                                                       FunctionCoefficient &p_fun, bool t_dependent)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
    p_fun(p_fun) {}
  
  SubsonicOutflowPBdrFaceIntegrator::SubsonicOutflowPBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       const ActiveGasModel &gasModel_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time, real_t p)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false),
    p(p), p_fun(std::function<real_t(const Vector&)>()) {}
  

void SubsonicOutflowPBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        if (t_dependent)
        {
            p_fun.SetTime(time);
        }
        p = p_fun.Eval(Tr, ip);
    }

    state2 = state1;
    Prandtl::PointStateView S1{state1.GetData()};
    Prandtl::PointStateViewRW S2{state2.GetData()};
    const real_t ke1 = gasModel.kinetic_energy_density(S1, thermoTables);
    const real_t ie = gasModel.internal_energy_from_pressure(S1, p, thermoTables);
    S2.set_energy(gasModel.L, ie + ke1);
}

void SubsonicOutflowPBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicOutflowPBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicOutflowPBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
