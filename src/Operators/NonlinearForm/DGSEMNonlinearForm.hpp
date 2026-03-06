#pragma once

#include "mfem.hpp"
#include "DGSEMIntegrator.hpp"
#include "BdrFaceIntegrator.hpp"
#include "general/forall.hpp"
#include "dgsem_cache.hpp"
#include <list>
#include <mpi.h>

namespace Prandtl
{
  
  // using namespace mfem;
  
  class DGSEMNonlinearForm : public ParNonlinearForm
  {
  public:

    Prandtl::DGSEMOperatorCache cache;
    Prandtl::DGSEMDeviceCache device_cache;
 
    void CreateOperatorCache();
    void GetOperatorCache(DGSEMOperatorCache &dgsem_operator_cache);
    void GetDeviceCache(DGSEMDeviceCache &dgsem_device_cache);
    void AssembleGeometricTerms();
    void AssembleElementVolumeGeometricTerms(mfem::ElementTransformation &);
    void AssembleFaceGeomCacheInterior();
    void SetDeviceCache(const DGSEMDeviceCache &dgsem_device_cache)
    {
      device_cache = dgsem_device_cache;
    }
    // Grab the face dof from the restriction (face,point) index
    // This answers: what is the facial dof that corresponds to
    // the facial point index for a given face in the restriction?
    int MapFp(int face_slot, int fp_restr) const
    {
      return cache.fqs_int->GetPermutedIndex(face_slot, fp_restr);
    }
    // And the inverse mapping
    int MapFpInv(int face_slot, int fp_perm) const {
      return cache.inv_fp_map[face_slot*cache.num_face_points + fp_perm];
    }
    DGSEMNonlinearForm(ParFiniteElementSpace *pfes);

    void MultLifting(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const;
    void MultLifting(const Vector &u, Vector &dudx, Vector &dudy) const;
    void MultLifting(const Vector &u, Vector &dudx) const;
    
    void Mult(const Vector &u, const Vector &dudx, const Vector &dudy, const Vector &dudz, Vector &dudt) const;
    void Mult(const Vector &u, const Vector &dudx, const Vector &dudy, Vector &dudt) const;
    void Mult(const Vector &u, const Vector &dudx, Vector &dudt) const;
    void Mult(const Vector &u, Vector &dudt) const;
    void MultOG(const Vector &u, Vector &dudt) const;
    real_t MultInviscid(const Vector &u, Vector &dudt) const;
    real_t MultVolumeInviscidDevice(const Vector &pu, Vector &pdudt) const;
    real_t MultInteriorFacesInviscidDevice(const Vector &pu, Vector &pdudt) const;
    void MultInteriorFacesInviscidHost(const Vector &pu, Vector &pdudt) const;
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
  private:
    mutable Vector aux2_x, aux2_y, aux2_z;
    Array<DGSEMIntegrator*> dnfi, fnfi;
    Array<BdrFaceIntegrator*> bfnfi;
    mutable ParGridFunction GRAD_X, GRAD_Y, GRAD_Z;
  };  
}
