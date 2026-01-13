#include "SubsonicInflowRVBdrFaceIntegrator.hpp"

namespace Prandtl
{

  SubsonicInflowRVBdrFaceIntegrator::SubsonicInflowRVBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       std::shared_ptr<const IdealGasModel> gasModel_,
                                                                       std::shared_ptr<const StateLayout> stateLayout_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time, FunctionCoefficient &rho_,
                                                                       VectorFunctionCoefficient &V_, bool t_dependent)
  : BdrFaceIntegrator(liftingScheme, gasModel_, stateLayout_, rsolver, Np, time, false, t_dependent),
    rho(rho_), V(V_) {}
  
  SubsonicInflowRVBdrFaceIntegrator::SubsonicInflowRVBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                       std::shared_ptr<const IdealGasModel> gasModel_,
                                                                       std::shared_ptr<const StateLayout> stateLayout_,
                                                                       const NumericalFlux &rsolver, const int Np,
                                                                       const real_t &time, real_t rho, const Vector &V)
  : BdrFaceIntegrator(liftingScheme, gasModel_, stateLayout_, rsolver, Np, time, true, false),
    r(rho), u(V), rho(std::function<real_t(const Vector&)>()), V(dim, std::function<void(const Vector&, Vector&)>())
  {
    u2 = 0.0;
    for (int idim = 0;idim < dim;idim++){
      u2 += u(idim)*u(idim);
    }
  }
  
  void SubsonicInflowRVBdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip)
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
    Prandtl::PointStateView S1{state1.GetData(), stateLayout.get()};
    Prandtl::PointStateViewRW S2{state2.GetData(), stateLayout.get()};
    S2.set_mass(r);
    S2.set_momentum(0, r * u(0));
    if (dim > 1) S2.set_momentum(1, r*u(1));
    if (dim > 2) S2.set_momentum(2, r*u(2));
    const real_t ke1 = gasModel->kinetic_energy_density(S1);
    const real_t ke2 = 0.5 * u2 * r;
    S2.set_energy(S1.energy()+ke2-ke1);
}

void SubsonicInflowRVBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, const Vector &dqdz_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = dqdz = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicInflowRVBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, const Vector &dqdy_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = dqdy = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void SubsonicInflowRVBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx_, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    dqdx = 0.0;
    fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

}
