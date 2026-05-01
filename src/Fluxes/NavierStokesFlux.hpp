#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"

namespace Prandtl
{
  
  // using namespace mfem;
  
  class NavierStokesFlux : public mfem::FluxFunction
  {
  private:
    const IdealGasModel gasModel;
  public:
    explicit NavierStokesFlux(const IdealGasModel &gasModel_)
      : mfem::FluxFunction(gasModel_.num_equations(), gasModel_.dim()), gasModel(gasModel_){};
    void ComputeViscousFlux(const mfem::Vector &state, const mfem::Vector &dqdx,
                            const mfem::Vector &dqdy, const mfem::Vector &dqdz,
                            mfem::DenseMatrix &flux) const;
    void ComputeViscousFlux(const mfem::Vector &state, const mfem::Vector &dqdx,
                            const mfem::Vector &dqdy, mfem::DenseMatrix &flux) const;
    void ComputeViscousFlux(const mfem::Vector &state, const mfem::Vector &dqdx,
                            mfem::DenseMatrix &flux) const;
    MFEM_HOST_DEVICE inline real_t pressure(const real_t *state) const
    {
      Prandtl::PointStateView S{state};
      return gasModel.pressure(S);
    }

    // These inviscid flux routines were lifted directly out of MFEM so
    // we can update them for gas models other than ideal single gas
    // (e.g. LTE, NLTE)
    /**
     * @brief Compute inviscid flux from conserved state
     *
     * @param state conserved state at current integration point
     * @param Tr current element transformation with the integration point
     * @param flux inviscid flux (ex, ideal single gas: F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)])
     * @return real_t maximum characteristic speed, c + |u| (c = speed of sound)
     */
    real_t ComputeFlux(const mfem::Vector &state,
                       mfem::ElementTransformation &Tr,
                       mfem::DenseMatrix &flux) const override;
    
    /**
     * @brief Compute inviscid flux along normal
     *
     * @param x conserved state at current integration point
     * @param normal normal vector, usually not a unit vector
     * @param Tr current element transformation with the integration point
     * @param fluxN inviscid flux dotted with normal
     * @return real_t maximum characteristic speed, c + |u.n|
     */
    real_t ComputeFluxDotN(const mfem::Vector &x,
                           const mfem::Vector &normal,
                           mfem::FaceElementTransformations &Tr,
                           mfem::Vector &fluxN) const override;

    template<typename GasT>
    MFEM_HOST_DEVICE inline
    static void ComputeViscousFluxKernel(const GasT &gas, const int dim,
                                         const real_t *state,
                                         const real_t *dprim_x,
                                         const real_t *dprim_y,
                                         const real_t *dprim_z,
                                         real_t visc_flux[Prandtl::MAXEQ][Prandtl::MAXDIM])
    {

      // TODO: Update for scalar transport

      // Zero the flux to start
      for(int q = 0;q < Prandtl::MAXEQ;q++){
        for(int idir = 0;idir < dim;idir++){
          visc_flux[q][idir] = 0.0;
        }
      }

      PointStateView S{state};
 
      // Access some physical constants
      const real_t mu = gas.viscosity(S);
      const real_t kappa = gas.thermal_conductivity(S);
      const real_t mu_bulk = gas.bulk_viscosity(S);

      // State structure constants
      const int eq_mass = gas.L.eq_mass;
      const int eq_mom0 = gas.L.eq_mom0;
      const int eq_ener = gas.L.eq_energy;
      const int nscalar = gas.L.num_scalars;
 
      // Make & populate gradient containers
      real_t grad_rho[Prandtl::MAXDIM] = {0.0, 0.0, 0.0};
      real_t grad_p[Prandtl::MAXDIM] = {0.0, 0.0, 0.0};
      real_t grad_t[Prandtl::MAXDIM] = {0.0, 0.0, 0.0};
      real_t grad_vel[Prandtl::MAXDIM][Prandtl::MAXDIM] = {{0.0}};

      grad_rho[0] = dprim_x[eq_mass];
      grad_vel[0][0] = dprim_x[eq_mom0];
      grad_p[0] = dprim_x[eq_ener];
      if(dim > 1){
        grad_rho[1] = dprim_y[eq_mass];
        grad_vel[0][1] = dprim_y[eq_mom0];
        grad_vel[1][0] = dprim_x[eq_mom0+1];
        grad_vel[1][1] = dprim_y[eq_mom0+1];
        grad_p[1] = dprim_y[eq_ener];
        if(dim > 2){
          grad_rho[2] = dprim_z[eq_mass];
          grad_vel[0][2] = dprim_z[eq_mom0];
          grad_vel[1][2] = dprim_z[eq_mom0+1];
          grad_vel[2][0] = dprim_x[eq_mom0+2];
          grad_vel[2][1] = dprim_y[eq_mom0+2];
          grad_vel[2][2] = dprim_z[eq_mom0+2];
          grad_p[2] = dprim_z[eq_ener];
        }
      }

      gas.grad_temperature(S, grad_rho, grad_p, grad_t);
      real_t vel[Prandtl::MAXDIM] = {0.0, 0.0, 0.0};
      real_t div_vel = 0.0;
      for(int i = 0;i < dim;i++){
        vel[i] = gas.velocity(S, i);
        div_vel += grad_vel[i][i];
      }

      // Build momentum/energy viscous fluxes by physical direction.
      // Output convention:
      //   flux_eq_dir[eq][dir]
      //
      // Mass row eq=0 remains zero.
      for (int dir = 0; dir < dim; ++dir)
        {
          // viscous stress tensor components tau[mom,dir]
          for (int mom = 0; mom < dim; ++mom)
            {
              real_t tau = 0.0;

              if (mom == dir)
                {
                  // Perserve legacy tau exactly
                  tau = mu * (2.0 * grad_vel[mom][dir] - mu_bulk * div_vel);
                }
              else
                {
                  tau = mu * (grad_vel[mom][dir] + grad_vel[dir][mom]);
                }
              
              visc_flux[eq_mom0 + mom][dir] = tau;
            }
          real_t eflux = kappa * grad_t[dir];
          for(int mom = 0;mom < dim;mom++)
            {
              eflux += vel[mom] * visc_flux[eq_mom0 + mom][dir];
            }
          visc_flux[eq_ener][dir] = eflux;
        }
    }

    template<typename GasT>
    MFEM_HOST_DEVICE inline
    static void compute_ref_viscous_flux(const GasT &gas,
                                         const int dim,
                                         const int neq,
                                         const real_t *state,
                                         const real_t *dqx,
                                         const real_t *dqy,
                                         const real_t *dqz,
                                         const real_t *adj_row,
                                         real_t *f_ref)
    {
      real_t flux_phys[Prandtl::MAXEQ][Prandtl::MAXDIM] = {{0.}};

      // Grab the physical flux
      ComputeViscousFluxKernel(gas, dim, state, dqx, dqy, dqz, flux_phys);
      
      for (int q = 0; q < neq; ++q)
        {
          f_ref[q] = 0.0;
          for (int j = 0; j < dim; ++j)
            f_ref[q] += adj_row[j] * flux_phys[q][j];
        }
    }
    
  };
  
}
