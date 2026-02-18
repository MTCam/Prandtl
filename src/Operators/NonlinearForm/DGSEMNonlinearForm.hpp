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
      // constants needed by kernels
      int p = 0;
      int dim = 0;
      int Np = 0;
      int Np_x = 0;
      int Np_y = 0;
      int Np_z = 0;
      int num_attr = 0;
      int num_elements = 0;
      int num_equations = 0;
      int ndof_scalar_el = 0;
      int num_face_points = 0;
      int num_interior_faces = 0;

      // Host Only: Integration rules, operators, restrictions 
      mfem::IntegrationRules GLIntRules{0, mfem::Quadrature1D::GaussLobatto};
      const IntegrationRule *ir = nullptr;
      const IntegrationRule *ir_face = nullptr;
      const IntegrationRule *ir_vol = nullptr;
      const mfem::ElementRestrictionOperator *restr_v = nullptr; // for volume vfes
      const mfem::FaceRestriction *restr_f = nullptr; // for face vfes (vector space)
      const mfem::FaceQuadratureSpace *fqs_int = nullptr; // interior faces perm

      // Data arrays for use on device
      mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
      mfem::Array<int> attr_marker;  // size nattr, 0/1
      mfem::Array<int> dnfi_marker;  // size nattr, 0/1
      mutable mfem::Vector elWaveSpeed; // size nelements
      mutable mfem::Vector ifWaveSpeed; // size ninterior faces
      mutable mfem::Vector bndWaveSpeed; // size nbnd faces
      mfem::Vector elJac;
      mfem::Vector elMetric;
      mfem::Vector D;
      mfem::Vector Dhat;
      mfem::Vector Dhat2;
      mfem::Vector face_normals;
      mfem::Vector face_wt_minus;
      mfem::Vector face_wt_plus;
    };
    OperatorCache cache;
    DGSEMDeviceCache device_cache;
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
    int MapFp(int face_id, int fp_restr) const
    {
      return cache.fqs_int->GetPermutedIndex(face_id, fp_restr);
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
