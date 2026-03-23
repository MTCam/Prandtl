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
  };
  
}
