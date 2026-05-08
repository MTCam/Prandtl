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


  template<typename PhysicsT>
  class NSOperator : public mfem::TimeDependentOperator,
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
    std::vector<std::shared_ptr<ParGridFunction> > grad_u;
    std::shared_ptr<ParGridFunction> r_gf;
    std::shared_ptr<Indicator> indicator;

    mutable Array<int> vdof_indices;
    mutable mfem::Vector el_vdofs, grad_vdofs;

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

    mutable IntegralMeasures diag0;
    mutable OperatorCache operator_cache;
    mutable DeviceCache device_cache;


  public:
    NSOperator(std::shared_ptr<ParFiniteElementSpace> vfes_,
               std::shared_ptr<ParFiniteElementSpace> fes0_,
               std::shared_ptr<ParMesh> pmesh_,
               std::shared_ptr<ParGridFunction> eta_,
               std::shared_ptr<ParGridFunction> alpha_,
               std::vector<std::shared_ptr<ParGridFunction> > &grad_u_,
               std::shared_ptr<Indicator> indicator_,
               const Gas &gasModel_,
               std::shared_ptr<ParGridFunction> r_gf_ = nullptr,
               const real_t alpha_max=0.5, const real_t alpha_min=0.001)
    : TimeDependentOperator(vfes_->GetTrueVSize()), ParNonlinearForm(vfes_.get()),
      vfes(vfes_), fes0(fes0_), pmesh(pmesh_),
      eta(eta_), grad_u(grad_u_),
      indicator(indicator_), num_equations(vfes->GetVDim()), dim(pmesh->SpaceDimension()),
      order(vfes->GetElementOrder(0)), num_elements(pmesh->GetNE()),
      Ndofs(vfes->GetFE(0)->GetDof()),
      modalThreshold(0.5 * std::pow(10.0, -1.8 * std::pow(order, 0.25))),
      r_gf(r_gf_), alpha_max(alpha_max), alpha_min(alpha_min),
      num_dofs_scalar(vfes_->GetTrueVSize()/vfes_->GetVDim())
    {
      operator_cache.gas = gasModel_;
      operator_cache.alpha = alpha_;
      operator_cache.entropyState.SetSize(vfes->GetVSize());

      diag0.mass = 0.0;
      diag0.ke = 0.0;
      diag0.en = 0.0;
    }

    void Finalize(real_t time)
    {
      this->SetTime(time);
      GetOperatorCache(vfes.get(), &operator_cache);
      AssembleBoundaryFaceGeometryTerms(vfes.get(), bdr_marker, &operator_cache);
#ifdef SUBCELL_FV_BLENDING
      ComputeSubcellMetrics(vfes.get(), &operator_cache);
#endif
      // Important that gasModel is POD for host<->device
      operator_cache.bc_descriptors = bc_descriptors;
      operator_cache.bc_scalar_data = bc_scalar_data;
      operator_cache.bc_vector_data = bc_vector_data;

      GetDeviceCache(operator_cache, device_cache);
    }

    ~NSOperator() {}

    inline real_t GetMaxCharSpeed()
    {
      return max_char_speed;
    }

    inline real_t& GetTimeRef()
    {
      return t;
    }

    void SetBCDescriptorData(const mfem::Array<Prandtl::BCDescriptor> &bc_descr, const mfem::Vector &bc_scalar_dat,
                             const mfem::Vector &bc_vector_dat)
    {
      bc_descriptors = bc_descr;
      bc_scalar_data = bc_scalar_dat;
      bc_vector_data = bc_vector_dat;
    }

    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker_)
    {
      bdr_marker.push_back(bdr_marker_);
    }

#ifdef SUBCELL_FV_BLENDING
    void ComputeBlendingCoefficient(const mfem::Vector &u) const;
    void ComputeBlendingCoefficientFromIndicator(const mfem::Vector &indicator_field) const;
    void ComputeIndicatorField(const mfem::Vector &u, mfem::Vector &indicator_field) const;
#endif

    void ComputeEntropyState(const mfem::Vector &u, mfem::Vector &e) const;
    void ComputeGradPrimFromGradEntropy(const mfem::Vector &u, std::vector<mfem::Vector *> &gradState) const;
    void ComputeIntegralMeasures(const mfem::Vector &u, IntegralMeasures &diag) const;
    void Mult(const Vector &u, Vector &dudt) const override;
    IntegralMeasures GetIntegralMeasuresBaseline() const { return diag0; }

    // Gradient Operator Interface (BR1 aux rhs)
    void GradOperator(const mfem::Vector &u, std::vector<mfem::Vector *> &grad_u) const;
    void GradOperator_Volume(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperator_InteriorFaces(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperator_BoundaryFaces(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;

    // NavierStokes RHS Interface
    real_t MultCNS(const mfem::Vector &u, const std::vector<mfem::Vector *> &grad_prim, mfem::Vector &dudt) const;
    real_t MultCNS_Volume(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                          mfem::Vector &pdudt) const;
    real_t MultCNS_InteriorFaces(const mfem::Vector &pu,
                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                 mfem::Vector &pdudt) const;
    real_t MultCNS_BoundaryFaces(const mfem::Vector &pu,
                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                 mfem::Vector &pdudt) const;
  };

}

#include "NSOperator_impl.hpp"
