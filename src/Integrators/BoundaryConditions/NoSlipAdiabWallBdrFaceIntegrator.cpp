#include "NoSlipAdiabWallBdrFaceIntegrator.hpp"

namespace Prandtl
{

NoSlipAdiabWallBdrFaceIntegrator::NoSlipAdiabWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, 
                                                                   const ActiveGasModel &gasModel_,
                                                                   const NumericalFlux &rsolver, const int Np,
                                                                   const real_t &time, FunctionCoefficient &qn_wall_,
                                                                   VectorFunctionCoefficient &V_wall_,bool t_dependent)
: SlipWallBdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
  qn_wall(qn_wall_), V_wall(V_wall_) {}
  
  NoSlipAdiabWallBdrFaceIntegrator::NoSlipAdiabWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                   const ActiveGasModel &gasModel_,
                                                                   const NumericalFlux &rsolver, const int Np,
                                                                   const real_t &time, real_t qn, const Vector &V)
  : SlipWallBdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false),
    qn(qn), V(V),
    qn_wall(std::function<real_t(const Vector&)>()), V_wall(dim, std::function<void(const Vector &, Vector&)>()) {}

void NoSlipAdiabWallBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        qn_wall.SetTime(time);
        V_wall.SetTime(time);

        qn = qn_wall.Eval(Tr, ip);
        V_wall.Eval(V, Tr, ip);
    }

    qn *= std::sqrt(nor * nor);

    fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, dqdz, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
    fluxN(en_eq) = V(0) * fluxN(mom_eq) + V(1) * fluxN(mom_eq+1) + V(2) * fluxN(mom_eq+2) + qn;
}

void NoSlipAdiabWallBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        qn_wall.SetTime(time);
        V_wall.SetTime(time);

        qn = qn_wall.Eval(Tr, ip);
        V_wall.Eval(V, Tr, ip);
    }

    qn *= std::sqrt(nor * nor);

    fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
    fluxN(en_eq) = V(0) * fluxN(mom_eq) + V(1) * fluxN(mom_eq+1) + qn;
}

void NoSlipAdiabWallBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        qn_wall.SetTime(time);
        V_wall.SetTime(time);

        qn = qn_wall.Eval(Tr, ip);
        V_wall.Eval(V, Tr, ip);
    }

    qn *= std::sqrt(nor * nor);

    fluxFunction.ComputeViscousFlux(state1, dqdx, thermoTables, flux_mat);
    flux_mat.Mult(nor, fluxN);
    fluxN(en_eq) = V(0) * fluxN(mom_eq) + qn;
}

void NoSlipAdiabWallBdrFaceIntegrator::ComputeBdrFaceLiftingFlux(const Vector &state1, Vector &fluxN, FaceElementTransformations &Tr, const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
    if (!constant)
    {
        V_wall.SetTime(time);
        V_wall.Eval(V, Tr, ip);        
    }
    Prandtl::PointStateView S{state1.GetData()};
    v = -gasModel.energy(S, thermoTables);
    fluxN(mass_eq) = gasModel.mass(S, thermoTables);
    fluxN(mom_eq) = V(0) * v;
    if (dim > 1)
    {
        fluxN(mom_eq+1) = V(1) * v;
        if (dim > 2)
        {
            fluxN(mom_eq+2) = V(2) * v;
        }
    }
    fluxN(en_eq) = -v;
    fluxN -= state1;
}

}
