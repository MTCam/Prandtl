#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"
#include "prandtl_device.hpp"

namespace Prandtl
{

  class ChandrashekarFlux : public NumericalFlux
  {
  private:
    mutable Vector metric;
    const IdealGasModel gasModel;
  public:
    ChandrashekarFlux(const NavierStokesFlux &fluxFunction, const IdealGasModel &gasModel_);
    real_t ComputeFaceFlux(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux) const override;
    real_t ComputeVolumeFlux(const Vector &state1, const Vector &state2, const Vector &metric1,
                             const Vector &metric2, Vector &F_tilde) override;
    
    // This is Riemann solver that computes the numerical flux for 2 point states
    // Will be ctx.iflux.ComputeVolumeFlux
    template<typename GasModelT>
    MFEM_HOST_DEVICE
    inline static real_t ComputeVolumeFluxKernel(const GasModelT &gasModel,
                                                 const real_t* q1,
                                                 const real_t* q2,
                                                 const real_t* met1,
                                                 const real_t* met2,
                                                 real_t* F_tilde)
    {
      const int dim = gasModel.dim();
      const int neq = gasModel.num_equations();
      
      // mean metric row
      real_t met[3] = {0,0,0};
      Kernels::ComputeMeanVec(met1, met2, met, dim);
      Prandtl::PointStateView S1{q1};
      Prandtl::PointStateView S2{q2};
      
      const real_t rho1 = gasModel.density(S1);
      const real_t rho2 = gasModel.density(S2);
      const real_t rho_ln = Kernels::ComputeLogMean(rho1, rho2, 1e-4);
      
      real_t mom_hat[3] = {0,0,0};
      real_t h_hat = 0;
      real_t vn = 0;
      real_t v2_1 = 0;
      real_t v2_2 = 0;
      
      for (int d=0; d<dim; ++d)
        {
          const real_t v1 = gasModel.velocity(S1, d);
          const real_t v2 = gasModel.velocity(S2, d);
          const real_t vbar = real_t(0.5)*(v1+v2);
          
          v2_1 += v1*v1;
          v2_2 += v2*v2;
          vn   += vbar * met[d];
          
          mom_hat[d] = rho_ln * vbar;
          
          h_hat += -real_t(0.25)*(v1*v1 + v2*v2) + vbar*vbar;
        }
      
      
      const real_t p1 = gasModel.pressure(S1);
      const real_t p2 = gasModel.pressure(S2);
      
      const real_t speed1 = Kernels::rsqrt(v2_1);
      const real_t speed2 = Kernels::rsqrt(v2_2);
      
      const real_t c1 = gasModel.sound_speed(S1);
      const real_t c2 = gasModel.sound_speed(S2);
      
      const real_t lambda_max = Kernels::rmax(speed1 + c1, speed2 + c2);
      
      // Single-component ideal-gas-specific KEPEC bits
      // TODO: Update/Craft KPEC fluxes for mixtures (and passive scalar components)
      const real_t beta1 = real_t(0.5) * rho1 / p1;
      const real_t beta2 = real_t(0.5) * rho2 / p2;
      const real_t beta_ln = Kernels::ComputeLogMean(beta1, beta2, 1e-4);
      
      const real_t p_hat = real_t(0.5) * (rho1 + rho2) / (beta1 + beta2);
      
      const real_t gm11 = gasModel.gamma(S1);
      const real_t gm12 = gasModel.gamma(S2);
      const real_t gm1_av_inv = real_t(2.0) / (gm11 + gm12 - real_t(2.0));
      
      h_hat += real_t(0.5) / beta_ln * gm1_av_inv + p_hat / rho_ln;
      
      // F_tilde layout: [rho, rhoV, rhoE]
      // NOTE: Caller *must* zero(or own) F_tilde (size: neq)
      // NOTE: HRM!  Why ZERO?  It appears that F_tilde is overwritten below
      F_tilde[0] = rho_ln * vn;
      for (int d=0; d<dim; ++d)
        {
          F_tilde[1 + d] = vn * mom_hat[d] + p_hat * met[d];
        }
      F_tilde[1 + dim] = rho_ln * vn * h_hat;
      
      // TODO: Updte for scalars, sigh
      // for (s=0; s<num_scalars; ++s) F_tilde[XXXX]= XXX
      
      return lambda_max;
    }
  };
 
  namespace Chandrashekar {
    // This is Riemann solver that computes the numerical flux for 2 point states
    // Will be ctx.iflux.ComputeVolumeFlux
    template<typename GasModelT>
    MFEM_HOST_DEVICE
    inline real_t ComputeVolumeFluxKernel(const GasModelT &gasModel,
                                          const real_t* q1,
                                          const real_t* q2,
                                          const real_t* met1,
                                          const real_t* met2,
                                          real_t* F_tilde)
    {
      const int dim = gasModel.dim();
      const int neq = gasModel.num_equations();
      
      // mean metric row
      real_t met[3] = {0,0,0};
      Kernels::ComputeMeanVec(met1, met2, met, dim);
      Prandtl::PointStateView S1{q1};
      Prandtl::PointStateView S2{q2};
      
      const real_t rho1 = gasModel.density(S1);
      const real_t rho2 = gasModel.density(S2);
      const real_t rho_ln = Kernels::ComputeLogMean(rho1, rho2, 1e-4);
      
      real_t mom_hat[3] = {0,0,0};
      real_t h_hat = 0;
      real_t vn = 0;
      real_t v2_1 = 0;
      real_t v2_2 = 0;
      
      for (int d=0; d<dim; ++d)
        {
          const real_t v1 = gasModel.velocity(S1, d);
          const real_t v2 = gasModel.velocity(S2, d);
          const real_t vbar = real_t(0.5)*(v1+v2);
          
          v2_1 += v1*v1;
          v2_2 += v2*v2;
          vn   += vbar * met[d];
          
          mom_hat[d] = rho_ln * vbar;
          
          h_hat += -real_t(0.25)*(v1*v1 + v2*v2) + vbar*vbar;
        }
      
      
      const real_t p1 = gasModel.pressure(S1);
      const real_t p2 = gasModel.pressure(S2);
      
      const real_t speed1 = Kernels::rsqrt(v2_1);
      const real_t speed2 = Kernels::rsqrt(v2_2);
      
      const real_t c1 = gasModel.sound_speed(S1);
      const real_t c2 = gasModel.sound_speed(S2);
      
      const real_t lambda_max = Kernels::rmax(speed1 + c1, speed2 + c2);
      
      // Single-component ideal-gas-specific KEPEC bits
      // TODO: Update/Craft KPEC fluxes for mixtures (and passive scalar components)
      const real_t beta1 = real_t(0.5) * rho1 / p1;
      const real_t beta2 = real_t(0.5) * rho2 / p2;
      const real_t beta_ln = Kernels::ComputeLogMean(beta1, beta2, 1e-4);
      
      const real_t p_hat = real_t(0.5) * (rho1 + rho2) / (beta1 + beta2);
      
      const real_t gm11 = gasModel.gamma(S1);
      const real_t gm12 = gasModel.gamma(S2);
      const real_t gm1_av_inv = real_t(2.0) / (gm11 + gm12 - real_t(2.0));
      
      h_hat += real_t(0.5) / beta_ln * gm1_av_inv + p_hat / rho_ln;
      
      // F_tilde layout: [rho, rhoV, rhoE]
      // NOTE: Caller *must* zero(or own) F_tilde (size: neq)
      // NOTE: HRM!  Why ZERO?  It appears that F_tilde is overwritten below
      F_tilde[0] = rho_ln * vn;
      for (int d=0; d<dim; ++d)
        {
          F_tilde[1 + d] = vn * mom_hat[d] + p_hat * met[d];
        }
      F_tilde[1 + dim] = rho_ln * vn * h_hat;
      
      // TODO: Updte for scalars, sigh
      // for (s=0; s<num_scalars; ++s) F_tilde[XXXX]= XXX
      
      return lambda_max;
    }
    
    struct InviscidFlux {
      MFEM_HOST_DEVICE inline real_t ComputeVolumeFlux(const IdealGasModel &gasModel,
                                                       const real_t *q1, const real_t *q2,
                                                       const real_t *met1, const real_t *met2,
                                                       real_t *F_tilde) const{
        return ComputeVolumeFluxKernel(gasModel, q1, q2, met1, met2, F_tilde); 
      }
    };
  }  
}
