#pragma once

#include "mfem.hpp"
#include "prandtl_device.hpp"
#include "DGSEMIntegrator.hpp"
#include "dgsem_device_cache.hpp"
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
    const Prandtl::DGSEM::OperatorCache *cache;
    Prandtl::DGSEM::DeviceCache device_cache;
    // Operator cache is a pointer because it is hefty, owned
    // by DGSEMOperator.   
    void SetOperatorCache(const Prandtl::DGSEM::OperatorCache *cache_){
      cache = cache_;
      cache->Output();
    }

    // And the inverse mapping
    int MapFpInv(int face_slot, int fp_perm) const {
      return cache->inv_fp_map[face_slot*cache->num_face_points + fp_perm];
    }
    // This function populates cache.face_is_shared
    // Also does some facial exploration to validate
    // Prandtl's assumptions about various face types.

    // Assumes: called after cache.restr_f->Mult(pu, u_faces);
    //
    // Requires: ir_face points match your restriction face-point ordering (nfp).
    //           (In DGSEM with GL points, this is typically true if you used the same rule.)
    // Args: prolongated state, restriction_faces
    void PopulateSharedStatePlusFromExchangeData(const mfem::Vector &pu,
                                                 mfem::Vector &u_faces) const
    {

      auto *mesh = fes->GetMesh();
      auto *pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
      auto *pmesh = dynamic_cast<ParMesh*>(mesh);
      
      MFEM_VERIFY(pfes, "pfes is null");
      MFEM_VERIFY(pmesh, "pmesh is null");
      const int nfp = cache->ir_face->GetNPoints();
      const int neq = pfes->GetVDim();

      // Make a ParGridFunction view of the prolonged vector 'pu'
      // so we can use MFEM's neighbor exchange + FaceNbrData() accessors.
      mfem::ParGridFunction X(pfes);
      // MakeRef wants non-const; we promise not to modify.
      mfem::Vector &pu_nc = const_cast<mfem::Vector&>(pu);
      X.MakeRef(pfes, pu_nc, 0);

      // Populate neighbor face data buffers (MFEM-managed packing/unpacking)
      X.ExchangeFaceNbrData();

      // We'll evaluate neighbor element states at face points using FE shapes
      mfem::Vector el_u2;
      mfem::Vector state2(neq);
      mfem::Vector shape2;

      mfem::Array<int> vdofs2;

      // Get a host-writable pointer to u_faces
      // This ok to do multiple times?
      real_t *uH = u_faces.HostReadWrite();

      // Index into u_faces
      auto idx_u_faces = [=](int face_slot, int side, int eq, int fp) -> int
      {
        return ((face_slot * 2 + side) * neq + eq) * nfp + fp;
      };

      const int n_shared_faces = pmesh->GetNSharedFaces();
      auto &shared_face_slot = cache->shared_to_interior_face;
      MFEM_VERIFY(shared_face_slot.Size() == n_shared_faces,
                  "shared_face_slot must have size == pmesh->GetNSharedFaces()");

      for (int sf = 0; sf < n_shared_faces; ++sf)
        {
          mfem::FaceElementTransformations *tr = pmesh->GetSharedFaceTransformations(sf, true);
          MFEM_VERIFY(tr, "GetSharedFaceTransformations returned null");

          // Neighbor element number in the face-neighbor numbering
          const int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();
          MFEM_VERIFY(Elem2NbrNo >= 0, "Bad Elem2NbrNo computed for shared face");

          const mfem::FiniteElement *fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);
          MFEM_VERIFY(fe2, "GetFaceNbrFE returned null");
          
          const int dof2 = fe2->GetDof();

          // Gather neighbor element dofs (for all eq) from MFEM's FaceNbrData() buffer
          pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

          // In DG, vdofs2.Size() should be dof2*neq for Ordering::byNODES;
          el_u2.SetSize(vdofs2.Size());
          X.FaceNbrData().GetSubVector(vdofs2, el_u2);

          // Interpret neighbor element dofs as a DenseMatrix with (dof2 x neq)
          const mfem::DenseMatrix el_u_mat2(el_u2.GetData(), dof2, neq);
          
          // Find the restriction face slot we need to patch for this shared face
          const int face_slot = shared_face_slot[sf];
          MFEM_VERIFY(face_slot >= 0, "shared_face_slot contains invalid entry");

          // Evaluate neighbor state at each face point and write into (+) side
          shape2.SetSize(dof2);

          for (int fp = 0; fp < nfp; ++fp)
            {
              const mfem::IntegrationPoint &ip = cache->ir_face->IntPoint(fp);
              int fp_restr = MapFpInv(face_slot, fp);
              // Set face integration point into both element transformations
              tr->SetAllIntPoints(&ip);

              // Note: for neighbor element on shared face, use Element2IntPoint
              fe2->CalcShape(tr->GetElement2IntPoint(), shape2);

              // state2 = sum_j el_u_mat2(j,eq) * shape2(j)  (for each eq)
              el_u_mat2.MultTranspose(shape2, state2);

              for (int eq = 0; eq < neq; ++eq)
                {
                  uH[idx_u_faces(face_slot, /*side=*/1, eq, fp_restr)] = state2(eq);
                }
            }
        }
    }
    void CreateOperatorCache();
    void GetOperatorCache(Prandtl::DGSEM::OperatorCache &dgsem_operator_cache);
    void GetDeviceCache(Prandtl::DGSEM::DeviceCache &dgsem_device_cache);
    void AssembleGeometricTerms();
    void AssembleElementVolumeGeometricTerms(mfem::ElementTransformation &);
    void AssembleFaceGeomCacheInterior();
    void SetDeviceCache(const Prandtl::DGSEM::DeviceCache &dcache_)
    {
      device_cache = dcache_;
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
