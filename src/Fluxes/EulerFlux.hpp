#pragma once

// Lifted from MFEM and gently massaged into place in Prandtl
#include "mfem.hpp"
#include "GasModel.hpp"

using mfem::DenseMatrix;
using mfem::ElementTransformation;
using mfem::FaceElementTransformations;
using mfem::FluxFunction;
using mfem::Vector;
using mfem::real_t;
using Prandtl::IdealGasModel;
using Prandtl::StateLayout;

namespace Prandtl
{
  
/// Inviscid
class EulerFlux : public FluxFunction
{
private:
  std::shared_ptr<const IdealGasModel> gasModel;
public:
  /**
   * @brief Construct a new EulerFlux FluxFunction with given spatial
   * dimension and specific heat ratio
   *
   * @param dim spatial dimension
   * @param specific_heat_ratio specific heat ratio, γ
   */
  explicit EulerFlux(std::shared_ptr<const StateLayout> stateLayout_, std::shared_ptr<const IdealGasModel> gasModel_)
    : FluxFunction(stateLayout_->nequations(), stateLayout_->dim), gasModel(std::move(gasModel_))
  { }


  /**
    * @brief Compute F(ρ, ρu, E)
    *
    * @param state state (ρ, ρu, E) at current integration point
    * @param Tr current element transformation with the integration point
    * @param flux F(ρ, ρu, E) = [ρuᵀ; ρuuᵀ + pI; uᵀ(E + p)]
    * @return real_t maximum characteristic speed, |u| + √(γp/ρ)
    */
   real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                      DenseMatrix &flux) const override;

   /**
    * @brief Compute normal flux, F(ρ, ρu, E)n
    *
    * @param x x (ρ, ρu, E) at current integration point
    * @param normal normal vector, usually not a unit vector
    * @param Tr current element transformation with the integration point
    * @param fluxN F(ρ, ρu, E)n = [ρu⋅n; ρu(u⋅n) + pn; (u⋅n)(E + p)]
    * @return real_t maximum characteristic speed, |u| + √(γp/ρ)
    */
   real_t ComputeFluxDotN(const Vector &x, const Vector &normal,
                          FaceElementTransformations &Tr,
                          Vector &fluxN) const override;
};
  
}
