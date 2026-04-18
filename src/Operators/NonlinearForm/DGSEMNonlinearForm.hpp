#pragma once

#include "mfem.hpp"
#include "DGSEMIntegrator.hpp"
#include "BdrFaceIntegrator.hpp"
#include "timer.hpp"
#include "general/forall.hpp"
#include "dgsem_cache_utilities.hpp"
#include "bc_kernels.hpp"

namespace Prandtl
{
  
  class DGSEMNonlinearForm : public mfem::ParNonlinearForm
  {
  private:
    mutable mfem::Vector aux2_x, aux2_y, aux2_z;
    mutable mfem::Vector rhs_aux_;
    mutable std::vector<mfem::Vector> grad_aux_;
    mutable std::vector<mfem::Vector> vol_grad_prim;
    mutable std::vector<mfem::Vector> int_grad_prim;
    mutable std::vector<mfem::Vector> bnd_grad_prim;
    mutable mfem::Vector vol_u;
    mutable mfem::Vector int_u;
    mutable mfem::Vector bnd_u;
    mfem::Array<DGSEMIntegrator*> dnfi, fnfi;
    mfem::Array<BdrFaceIntegrator*> bfnfi;
    mutable mfem::ParGridFunction GRAD_X, GRAD_Y, GRAD_Z;
    Prandtl::DGSEMOperatorCache *cache = nullptr;
    Prandtl::DGSEMDeviceCache device_cache;
    
  public:
    DGSEMNonlinearForm(mfem::ParFiniteElementSpace *pfes);
    void SetOperatorCache(DGSEMOperatorCache *cache_){
      cache = cache_;
      GetDeviceCache(*cache, device_cache);
    }
    real_t MultCNS(const mfem::Vector &u, const std::vector<mfem::Vector *> &grad_prim, mfem::Vector &dudt) const;
    real_t MultCNS_Volume(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                          mfem::Vector &pdudt) const;
    real_t MultCNS_InteriorFaces(const mfem::Vector &pu,
                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                 mfem::Vector &pdudt) const;
    real_t MultCNS_BoundaryFaces(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                                 mfem::Vector &pdudt) const;
    void MultLifting(const mfem::Vector &u, mfem::Vector &dudx,
                     mfem::Vector &dudy, mfem::Vector &dudz) const;
    void MultLifting(const mfem::Vector &u, mfem::Vector &dudx, mfem::Vector &dudy) const;
    void GradOperator(const mfem::Vector &u, std::vector<mfem::Vector *> &grad_u) const;
    void GradOperatorVolumeDevice(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperatorVolumeHost(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperatorInteriorFacesDevice(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperatorInteriorFacesHost(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperatorBoundaryFacesDevice(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperatorBoundaryFacesDeviceNOOP(const mfem::Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void GradOperatorBoundaryFacesHost(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const;
    void MultLifting(const mfem::Vector &u, mfem::Vector &dudx) const;

    void Mult(const mfem::Vector &u, const mfem::Vector &dudx, const mfem::Vector &dudy,
              const mfem::Vector &dudz, mfem::Vector &dudt) const;
    void Mult(const mfem::Vector &u, const mfem::Vector &dudx,
              const mfem::Vector &dudy, mfem::Vector &dudt) const;
    void Mult(const mfem::Vector &u, const mfem::Vector &dudx, mfem::Vector &dudt) const;
    real_t MultBndFacesInviscidDevice(const Vector &pu, Vector &pdudt) const;
    real_t MultBndFacesInviscidHost(const Vector &pu, Vector &pdudt) const;
    void Mult(const mfem::Vector &u, mfem::Vector &dudt) const;
  
    real_t MultVolumeInviscidDevice(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultInteriorFacesInviscidDevice(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    real_t MultEuler(const mfem::Vector &pu, mfem::Vector &pdudt) const;
    void AddDomainIntegrator(DGSEMIntegrator *nlfi)
    {
      dnfi.Append(nlfi);
      dnfi_marker.Append(NULL);
    }

    void AddInteriorFaceIntegrator(DGSEMIntegrator *nlfi)
    {
      fnfi.Append(nlfi);
    }
    
    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, mfem::Array<int> &bdr_marker)
    {
      bfnfi.Append(bfi);
      bfnfi_marker.Append(&bdr_marker);
    }

  };
}
