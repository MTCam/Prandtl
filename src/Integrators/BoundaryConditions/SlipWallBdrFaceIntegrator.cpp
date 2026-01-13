#include "SlipWallBdrFaceIntegrator.hpp"
#include "BasicOperations.hpp"
#include "Flow.hpp"

namespace Prandtl
{

  SlipWallBdrFaceIntegrator::SlipWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme,
                                                       std::shared_ptr<const IdealGasModel> gasModel_,
                                                       std::shared_ptr<const StateLayout> stateLayout_, 
                                                       const NumericalFlux &rsolver, int Np, const real_t &time,
                                                       bool constant, bool t_dependent)
    : BdrFaceIntegrator(liftingScheme, gasModel_, stateLayout_, rsolver, Np, time, constant, t_dependent)
{
    unit_nor.SetSize(rsolver.GetFluxFunction().dim);
    prim.SetSize(rsolver.GetFluxFunction().num_equations);
}


real_t SlipWallBdrFaceIntegrator::ComputeBdrFaceInviscidFlux(const Vector &state1, Vector &state2,
    Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip)
{
    unit_nor = nor;
    Normalize(unit_nor);
    state2 = state1;
    RotateState(state2, unit_nor, *stateLayout);
    Prandtl::PointStateView S{state2.GetData(), stateLayout.get()};
    const real_t p_star = Prandtl::Flow::slipwall_pstar(S, *gasModel);
    const real_t v = S.velocity(0);
    const real_t c = gasModel->sound_speed(S);
    fluxN = 0.0;
    fluxN(mom_eq) = p_star * nor(0);
    if (dim > 1){
      fluxN(mom_eq+1) = p_star * nor(1);
      if (dim > 2)
        fluxN(mom_eq+2) = p_star * nor(2);
    }
    return std::abs(v) + c;
}

}
