#include "NoSlipIsothWallBdrFaceIntegrator.hpp"
#include "Flow.hpp"

namespace Prandtl
{

NoSlipIsothWallBdrFaceIntegrator::NoSlipIsothWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                   std::shared_ptr<const IdealGasModel> gasModel_,
                                                                   const NumericalFlux &rsolver, const int Np,
                                                                   const real_t &time,
                                                                   FunctionCoefficient &T_wall_, VectorFunctionCoefficient &V_wall_,
                                                                   bool t_dependent)
: SlipWallBdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, false, t_dependent),
  T_wall(T_wall_), V_wall(V_wall_) {}
  
NoSlipIsothWallBdrFaceIntegrator::NoSlipIsothWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                                   std::shared_ptr<const IdealGasModel> gasModel_,
                                                                   const NumericalFlux &rsolver, const int Np,
                                                                   const real_t &time, real_t &T, const Vector &V)
: SlipWallBdrFaceIntegrator(liftingScheme, gasModel_, rsolver, Np, time, true, false),
  T(T), V(V), T_wall(std::function<real_t(const Vector&)>()), V_wall(dim, std::function<void(const Vector&, Vector&)>())
{}
  
void NoSlipIsothWallBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, dqdz, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void NoSlipIsothWallBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void NoSlipIsothWallBdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    fluxFunction.ComputeViscousFlux(state1, dqdx, flux_mat);
    flux_mat.Mult(nor, fluxN);
}

void NoSlipIsothWallBdrFaceIntegrator::ComputeBdrFaceLiftingFlux(const Vector &state1, Vector &fluxN, FaceElementTransformations &Tr,
                                                                 const IntegrationPoint &ip)
{
    if (!constant)
    {
        T_wall.SetTime(time);
        V_wall.SetTime(time);

        T = T_wall.Eval(Tr, ip);
        V_wall.Eval(V, Tr, ip);
    }
    Prandtl::PointStateView Se{state1.GetData()};
    const real_t beta = Prandtl::Flow::isothermal_wall_beta(Se, T, *gasModel);
    fluxN(mass_eq) = gasModel->mass(Se);
    fluxN(mom_eq) = V(0) * beta;
    if (dim > 1)
    {
        fluxN(mom_eq+1) = V(1) * beta;
        if (dim > 2)
        {
            fluxN(mom_eq+2) = V(2) * beta;
        }
    }
    fluxN(en_eq) = -beta;
    fluxN -= state1;
}

}
