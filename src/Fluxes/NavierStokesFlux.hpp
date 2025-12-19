#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"

namespace Prandtl
{

using namespace mfem;

class NavierStokesFlux : public EulerFlux
{
private:
    mutable real_t div, cv_dTdx, cv_dTdy, cv_dTdz, lambda, mu;
    mutable Vector prim;
    const real_t gamma, gammaM1, gammaM1Inverse;
    const real_t PrInverse;
    const real_t mu_bulk, mu0, R_gas, Ts, T0, T0pTs;
    std::shared_ptr<const IdealGasModel> gasModel;
    std::shared_ptr<const StateLayout> stateLayout;
public:
  explicit NavierStokesFlux(std::shared_ptr<const StateLayout> stateLayout_, std::shared_ptr<const IdealGasModel> gasModel_)
        : EulerFlux(stateLayout_->dim, gasModel_->eos.phys->gamma),
          gamma(gasModel_->eos.phys->gamma), gammaM1(gasModel_->eos.phys->gammaM1), gammaM1Inverse(gasModel_->eos.phys->gammaM1Inverse),
          PrInverse(gasModel_->eos.phys->PrInverse),
          mu(gasModel_->eos.phys->mu), mu0(gasModel_->eos.phys->mu0), mu_bulk(gasModel_->eos.phys->mu_bulk),
          R_gas(gasModel_->eos.phys->R_gas),
          Ts(gasModel_->eos.phys->Ts), T0(gasModel_->eos.phys->T0), T0pTs(gasModel_->eos.phys->T0 + gasModel_->eos.phys->Ts),
          stateLayout(std::move(stateLayout_)), gasModel(std::move(gasModel_))
    {
        prim.SetSize(stateLayout->dim + 2);
    }
    real_t ComputeInviscidFlux(const Vector &state, ElementTransformation &Tr, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, DenseMatrix &flux) const;
    real_t ComputeInviscidFluxDotN(const Vector &x, const Vector &nor, FaceElementTransformations &Tr, Vector &fluxN) const;

  // inline real_t ComputeViscosity(real_t rho, real_t p)
  //  {
  //      real_t T = p / (rho * R_gas);
  //      return mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
  //  }
};

}
