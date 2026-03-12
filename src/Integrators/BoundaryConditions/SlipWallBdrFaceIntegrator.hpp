#pragma once

#include "BdrFaceIntegrator.hpp"
#include "Flow.hpp"

namespace Prandtl

{

using namespace mfem;

class SlipWallBdrFaceIntegrator : public BdrFaceIntegrator
{
private:
    Vector unit_nor;
    Vector prim;
    real_t p_star, an;
public:
  SlipWallBdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, const IdealGasModel &gasModel_,
                            const NumericalFlux &rsolver, int Np, const real_t &time, bool constant = true, bool t_dependent = false);
    virtual real_t ComputeBdrFaceInviscidFlux(const Vector &state1, Vector &state2, Vector &fluxN, const Vector &nor,
                                              FaceElementTransformations &Tr, const IntegrationPoint &ip) override;
};

  template<typename GasModelT>
  MFEM_HOST_DEVICE
  real_t SlipWallInviscidFluxKernel(GasModelT gasModel, const real_t *state1, const real_t *nor, real_t *fluxN)
  {
    // TODO: use MAXEQ instead of 5
    real_t unit_nor[3];
    real_t state2[5];
    const int dim = gasModel.L.dim;
    const int neq = gasModel.L.nequations();
    for(int idim = 0;idim < dim;idim++)
      unit_nor[idim] = nor[idim];
    for(int ieq = 0;ieq < neq;ieq++){
      state2[ieq] = state1[ieq];
      fluxN[ieq] = 0.0;
    }
    Prandtl::Kernels::Normalize(dim, unit_nor);
    Prandtl::Flow::RotateState(gasModel.L, state2, unit_nor);
    Prandtl::PointStateView S{state2};
    const real_t p_star = Prandtl::Flow::slipwall_pstar(S, gasModel);
    const real_t v = gasModel.velocity(S, 0);
    const real_t c = gasModel.sound_speed(S);
    const int mom_eq = gasModel.L.eq_mom0;

    fluxN[mom_eq] = p_star * nor[0];
    if (dim > 1){
      fluxN[mom_eq+1] = p_star * nor[1];
      if (dim > 2)
        fluxN[mom_eq+2] = p_star * nor[2];
    }
    return std::abs(v) + c;
  }

}
