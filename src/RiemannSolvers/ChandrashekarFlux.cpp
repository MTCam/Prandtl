#include "ChandrashekarFlux.hpp"
#include "BasicOperations.hpp"


namespace Prandtl
{

  ChandrashekarFlux::ChandrashekarFlux(const NavierStokesFlux &fluxFunction, const IdealGasModel &gasModel_)
    : NumericalFlux(fluxFunction), gasModel(gasModel_)
  {
    metric.SetSize(dim);
  }
  
  real_t ChandrashekarFlux::ComputeVolumeFlux(const Vector &state1, const Vector &state2,
                                              const Vector &metric1, const Vector &metric2,
                                              Vector &F_tilde)
  {
    return Chandrashekar::ComputeVolumeFluxKernel(gasModel, state1.GetData(), state2.GetData(),
                                                  metric1.GetData(), metric2.GetData(),
                                                  F_tilde.GetData());
  }


real_t ChandrashekarFlux::ComputeFaceFlux(const Vector &state1, const Vector &state2,
                                          const Vector &nor, Vector &flux) const
{
    PointStateView S1{state1.GetData()};
    PointStateView S2{state2.GetData()};
    
    const real_t nor_mag = nor.Norml2();

    const real_t rho1 = gasModel.density(S1);
    const real_t rho2 = gasModel.density(S2);
    const real_t rho_mean = 0.5 * (rho1 + rho2);
    const real_t rho_ln = ComputeLogMean(rho1, rho2);
    const real_t drho = rho2 - rho1;
    real_t mom[3] = {0.0, 0.0, 0.0};
    real_t mom1[3] = {0.0, 0.0, 0.0};
    real_t mom2[3] = {0.0, 0.0, 0.0};
    real_t hhat = 0.0;
    real_t diss = 0.0;
    real_t v21 = 0.0;
    real_t v22 = 0.0;
    real_t vn = 0.0;
    for(int idim = 0;idim < dim;idim++){
      mom1[idim] = gasModel.momentum(S1, idim);
      mom2[idim] = gasModel.momentum(S2, idim);
      const real_t v1 = mom1[idim]/rho1;
      const real_t v2 = mom2[idim]/rho2;
      const real_t vbar = 0.5 * (v1 + v2);
      const real_t dv = v2 - v1;
      v21 += v1*v1;
      v22 += v2*v2;
      vn += vbar * nor(idim);
      mom[idim] = rho_ln * vbar;
      hhat += -0.25 * (v1*v1 + v2*v2) + vbar * vbar;
      diss += 0.5 * drho * v1*v2 + rho_mean * dv * vbar;
    }

    const real_t p1 = gasModel.pressure(S1);
    const real_t p2 = gasModel.pressure(S2);

    const real_t vmag1 = std::sqrt(v21);
    const real_t vmag2 = std::sqrt(v22);

    const real_t c1 = gasModel.sound_speed(S1);
    const real_t c2 = gasModel.sound_speed(S2);

    const real_t lambda_max = std::max(vmag1+c1, vmag2+c2);

    const real_t beta1 = 0.5 * rho1 / p1;
    const real_t beta2 = 0.5 * rho2 / p2;
    const real_t beta_ln = ComputeLogMean(beta1, beta2);

    const real_t p_hat = 0.5 * (rho1 + rho2) / (beta1 + beta2);

    // Use the average gamma for now
    // TODO: Craft KEPEC fluxes for LTE/NLTE
    const real_t gm11 = gasModel.gamma(S1);
    const real_t gm12 = gasModel.gamma(S2); 
    const real_t gm1_av_inv = 2.0/(gm11 + gm12 - 2.0);

    hhat += 0.5 / beta_ln * gm1_av_inv + p_hat / rho_ln;
    diss += 0.5 * drho * gm1_av_inv / beta_ln + 0.5 * rho_mean * gm1_av_inv * (1.0 / beta2 - 1.0 / beta1);

    flux(0) = rho_ln * vn - 0.5 * lambda_max * (rho2 - rho1) * nor_mag;
    for (int d = 0; d < dim; d++)
    {
      flux(1 + d) = vn * mom[d] + p_hat * nor(d) - 0.5 * lambda_max * (mom2[d]-mom1[d]) * nor_mag;
    }
    flux(1 + dim) = rho_ln * vn * hhat - 0.5 * lambda_max * diss * nor_mag;

    return lambda_max;
}

}
