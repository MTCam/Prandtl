#pragma once

#include "DGSEMIntegrator.hpp"
#include "DGSEMNonlinearForm.hpp"
#include "BdrFaceIntegrator.hpp"
#include "ModalBasis.hpp"
#include "Indicator.hpp"
#include "BasicOperations.hpp"
#include "GasModel.hpp"
#include "dgsem_cache_utilities.hpp"
#include "bc_cache_utilities.hpp"

namespace Prandtl
{
  // using namespace mfem;
  template<typename PhysicsT>
  class EulerOperator : public mfem::TimeDependentOperator,
                        public mfem::ParNonlinearForm
  {

  public:
    using Physics = PhysicsT;
    using OperatorCache = DGSEMOperatorCacheT<Physics>;
    using DeviceCache = DGSEMDeviceCacheT<Physics>;
    using Gas = typename Physics::GasModel;
    using InviscidFlux = typename Physics::InviscidFlux;

  private:
    std::shared_ptr<ParFiniteElementSpace> vfes;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::shared_ptr<ParMesh> pmesh;
    std::shared_ptr<ParGridFunction> eta;
    std::shared_ptr<ParGridFunction> r_gf;
    std::shared_ptr<Indicator> indicator;

    const int num_equations, dim, order, num_elements;
    const int num_dofs_scalar;
    const int Ndofs;

    const real_t sharpness_fac = 9.21024;
    const real_t modalThreshold;
    const real_t alpha_min;
    const real_t alpha_max;

    mutable real_t max_char_speed;

    std::vector<Array<int>> bdr_marker;
    mfem::Array<Prandtl::BCDescriptor> bc_descriptors;
    mfem::Vector bc_vector_data;
    mfem::Vector bc_scalar_data;

    mutable real_t alpha_dof;
    mutable IntegralMeasures diag0;

    mutable OperatorCache operator_cache;
    mutable DeviceCache device_cache;

  public:

    EulerOperator(std::shared_ptr<mfem::ParFiniteElementSpace> vfes_,
                  std::shared_ptr<mfem::ParFiniteElementSpace> fes0_,
                  std::shared_ptr<mfem::ParMesh> pmesh_,
                  std::shared_ptr<mfem::ParGridFunction> eta_,
                  std::shared_ptr<mfem::ParGridFunction> alpha_,
                  std::shared_ptr<Indicator> indicator_,
                  const Gas &gasModel_,
                  std::shared_ptr<ParGridFunction> r_gf_ = nullptr,
                  const real_t alpha_max = 0.5, const real_t alpha_min = 0.001)
    : mfem::TimeDependentOperator(vfes_->GetTrueVSize()),
      mfem::ParNonlinearForm(vfes_.get()),
      vfes(vfes_), fes0(fes0_), pmesh(pmesh_),
      eta(eta_), indicator(indicator_),
      num_equations(vfes->GetVDim()), dim(pmesh->SpaceDimension()),
      order(vfes->GetElementOrder(0)), num_elements(pmesh->GetNE()),
      Ndofs(vfes->GetFE(0)->GetDof()),
      modalThreshold(0.5 * std::pow(10.0, -1.8 * std::pow(order, 0.25))),
      r_gf(r_gf_), alpha_max(alpha_max), alpha_min(alpha_min),
      num_dofs_scalar(vfes_->GetTrueVSize()/vfes_->GetVDim())
    {
      operator_cache.gas = gasModel_;
      operator_cache.alpha = alpha_;
      diag0.mass = 0.0;
      diag0.ke = 0.0;
      diag0.en = 0.0;
    }
    
    ~EulerOperator();

    void SetBCDescriptorData(const mfem::Array<Prandtl::BCDescriptor> &bc_descr, const mfem::Vector &bc_scalar_dat,
                             const mfem::Vector &bc_vector_dat)
    {
      bc_descriptors = bc_descr;
      bc_scalar_data = bc_scalar_dat;
      bc_vector_data = bc_vector_dat;
    }

    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker);
    void Finalize(real_t time=0);

#ifdef SUBCELL_FV_BLENDING
    void ComputeBlendingCoefficient(const Vector &u) const;
    void ComputeBlendingCoefficientFromIndicator(const Vector &indicator_field) const;
    void ComputeIndicatorField(const Vector &u, Vector &indicator_field) const;
#endif

    void ComputeIntegralMeasures(const Vector &u, IntegralMeasures &diag) const;
    IntegralMeasures GetIntegralMeasuresBaseline() const { return diag0; }

    inline real_t GetMaxCharSpeed()
    {
      return max_char_speed;
    }

    inline real_t& GetTimeRef()
    {
      return t;
    }

    void Mult(const Vector &u, Vector &dudt) const override;
    real_t MultEuler(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultEuler_Volume(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultEuler_InteriorFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultEuler_BoundaryFaces(const Vector &pu, Vector &pdudt) const;

  };
  
}

#include "EulerOperator_impl.hpp"
