#pragma once

#include "mfem.hpp"
#include "prandtl_device.hpp"
#include "DGSEMIntegrator.hpp"
#include "dgsem_device_cache.hpp"
#include "BdrFaceIntegrator.hpp"
#include "general/forall.hpp"
#include "dgsem_device_cache.hpp"

namespace Prandtl
{
  
  // using namespace mfem;
  
  class DGSEMNonlinearForm : public ParNonlinearForm
  {
  public:
    struct OperatorCache {
      int dim = 0;
      int num_elements = 0;
      int num_attr = 0;
      int ndof_scalar_el = 0;
      int num_equations = 0;
      mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
      mfem::Array<int> attr_marker;  // size nattr, 0/1
      mfem::Array<int> dnfi_marker;  // size nattr, 0/1
      mutable mfem::Vector elWaveSpeed; // size nelements
      const mfem::ElementRestrictionOperator *restr_v = nullptr; // for vfes (vector space)
      mfem::Vector elJac;
      mfem::Vector elMetric;
      mfem::Vector D;
      mfem::Vector Dhat;
      mfem::Vector Dhat2;
    };
    OperatorCache cache;
    Prandtl::DGSEMDeviceCache device_cache;

    DGSEMNonlinearForm(ParFiniteElementSpace *pfes);
    void SetDeviceCache(const Prandtl::DGSEMDeviceCache &dgsem_device_cache)
    {
      device_cache = dgsem_device_cache;
    }
    void MultLifting(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const;
    void MultLifting(const Vector &u, Vector &dudx, Vector &dudy) const;
    void MultLifting(const Vector &u, Vector &dudx) const;
    
    void Mult(const Vector &u, const Vector &dudx, const Vector &dudy, const Vector &dudz, Vector &dudt) const;
    void Mult(const Vector &u, const Vector &dudx, const Vector &dudy, Vector &dudt) const;
    void Mult(const Vector &u, const Vector &dudx, Vector &dudt) const;
    void Mult(const Vector &u, Vector &dudt) const;
    void MultOG(const Vector &u, Vector &dudt) const;
    real_t MultInviscid(const Vector &u, Vector &dudt) const;
    real_t MultInviscidVolumeDevice(const Vector &pu, Vector &pdudt) const;
    real_t MultInviscidVolumeHost(const Vector &pu, Vector &pdudt) const;
    real_t MultInviscidVolumeHost2(const Vector &pu, Vector &pdudt) const;
    void AddDomainIntegrator(DGSEMIntegrator *nlfi)
    {
        dnfi.Append(nlfi);
        dnfi_marker.Append(NULL);
    }
         
    void AddInteriorFaceIntegrator(DGSEMIntegrator *nlfi)
    {
        fnfi.Append(nlfi);
    }


    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker)
    {
        bfnfi.Append(bfi);
        bfnfi_marker.Append(&bdr_marker);
    }

  void CreateOperatorCache();
  void GetOperatorCache(Prandtl::DGSEMOperatorCache &dgsem_operator_cache);
  void GetDeviceCache(Prandtl::DGSEMDeviceCache &dgsem_device_cache);
  private:
    mutable Vector aux2_x, aux2_y, aux2_z;
    Array<DGSEMIntegrator*> dnfi, fnfi;
    Array<BdrFaceIntegrator*> bfnfi;
    mutable ParGridFunction GRAD_X, GRAD_Y, GRAD_Z;
  };  
}
