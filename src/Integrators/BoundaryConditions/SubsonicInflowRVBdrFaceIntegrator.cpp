#include "SubsonicInflowRVBdrFaceIntegrator.hpp"

namespace Prandtl
{

  SubsonicInflowRVBdrFaceIntegrator::SubsonicInflowRVBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       const ActiveGasModel &gasModel_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time, FunctionCoefficient &rho_,
                                                                       VectorFunctionCoefficient &V_, bool t_dependent)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
    rho(rho_), V(V_) {}
  
  SubsonicInflowRVBdrFaceIntegrator::SubsonicInflowRVBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       const ActiveGasModel &gasModel_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time, real_t rho, const Vector &V)
  : BdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false),
    r(rho), u(V), rho(std::function<real_t(const Vector&)>()), V(dim, std::function<void(const Vector&, Vector&)>())
  {
    u2 = 0.0;
    for (int idim = 0;idim < dim;idim++){
      u2 += u(idim)*u(idim);
    }
  }
  
  void SubsonicInflowRVBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        if (t_dependent)
        {
            rho.SetTime(time);
            V.SetTime(time);
        }
        r = rho.Eval(Tr, ip);
        V.Eval(u, Tr, ip);
        u2 = u(0)*u(0);
        if (dim > 1) u2 += u(1)*u(1);
        if (dim > 2) u2 += u(2)*u(2);
    }
    Prandtl::PointStateView S1{state1.GetData()};
    Prandtl::PointStateViewRW S2{state2.GetData()};
    S2.set_mass(gasModel.L, r);
    S2.set_momentum(gasModel.L, 0, r * u(0));
    if (dim > 1) S2.set_momentum(gasModel.L, 1, r*u(1));
    if (dim > 2) S2.set_momentum(gasModel.L, 2, r*u(2));
    const real_t ke1 = gasModel.kinetic_energy_density(S1, thermoTables);
    const real_t ke2 = 0.5 * u2 * r;
    S2.set_energy(gasModel.L, S1.energy(gasModel.L)+ke2-ke1);
}

void SubsonicInflowRVBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicInflowRVBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicInflowRVBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
