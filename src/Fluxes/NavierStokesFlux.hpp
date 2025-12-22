#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"

namespace Prandtl
{

using namespace mfem;

class NavierStokesFlux : public FluxFunction
{
private:
    std::shared_ptr<const IdealGasModel> gasModel;
    std::shared_ptr<const StateLayout> stateLayout;
public:
  explicit NavierStokesFlux(std::shared_ptr<const StateLayout> stateLayout_, std::shared_ptr<const IdealGasModel> gasModel_)
    : FluxFunction(stateLayout_->nequations(), stateLayout_->dim),
      stateLayout(std::move(stateLayout_)), gasModel(std::move(gasModel_))
    {}
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, DenseMatrix &flux) const;
    void ComputeViscousFlux(const Vector &state, const Vector &dqdx, DenseMatrix &flux) const;

   /**
    * @brief Compute inviscid flux from conserved state
    *
    * @param state conserved state at current integration point
    * @param Tr current element transformation with the integration point
    * @param flux inviscid flux (ex, ideal single gas: F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)])
    * @return real_t maximum characteristic speed, c + |u| (c = speed of sound)
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute inviscid flux along normal
    *
    * @param x conserved state at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxN inviscid flux dotted with normal
    * @return real_t maximum characteristic speed, c + |u.n|
    */
   real_t ComputeFluxDotN(const Vector &x, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};

}
