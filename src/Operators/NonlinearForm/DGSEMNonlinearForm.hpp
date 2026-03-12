#pragma once

#include "mfem.hpp"
#include "DGSEMIntegrator.hpp"
#include "BdrFaceIntegrator.hpp"
#include "timer.hpp"
#include "general/forall.hpp"
#include "dgsem_cache_utilities.hpp"

namespace Prandtl
{
  
  class DGSEMNonlinearForm : public mfem::ParNonlinearForm
  {
  private:
    mutable mfem::Vector aux2_x, aux2_y, aux2_z;
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
    void MultLifting(const mfem::Vector &u, mfem::Vector &dudx,
                     mfem::Vector &dudy, mfem::Vector &dudz) const;
    void MultLifting(const mfem::Vector &u, mfem::Vector &dudx, mfem::Vector &dudy) const;
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
    real_t MultInviscid(const mfem::Vector &pu, mfem::Vector &pdudt) const;
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
