#pragma once

#include "mfem.hpp"

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

public:
    NavierStokesFlux(const int dim, real_t gamma_, real_t Pr, real_t mu_, real_t mu0, real_t mu_bulk, real_t R_gas, real_t Ts, real_t T0)
        : EulerFlux(dim, gamma_), gamma(gamma_), gammaM1(gamma - 1.0), gammaM1Inverse(1.0 / gammaM1), PrInverse(1.0 / Pr),
          mu(mu_), mu0(mu0), mu_bulk(mu_bulk), R_gas(R_gas), Ts(Ts), T0(T0), T0pTs(T0 + Ts)
    {
      prim.SetSize(dim + 2);
    }
    real_t ComputeInviscidFlux(const Vector &state, ElementTransformation &Tr, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, DenseMatrix &flux) const;
    real_t ComputeInviscidFluxDotN(const Vector &x, const Vector &nor, FaceElementTransformations &Tr, Vector &fluxN) const;

    inline real_t ComputeViscosity(real_t rho, real_t p)
    {
        real_t T = p / (rho * R_gas);
        return mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
    }
};

}
