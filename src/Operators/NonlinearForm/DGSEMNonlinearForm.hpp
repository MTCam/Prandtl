#pragma once

#include "mfem.hpp"
#include "prandtl_device.hpp"
#include "DGSEMIntegrator.hpp"
#include "dgsem_device_cache.hpp"
#include "BdrFaceIntegrator.hpp"
#include "general/forall.hpp"
#include "dgsem_device_cache.hpp"
#include <list>
#include <mpi.h>

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

      // Aux data for preprocessing
      mfem::Array<char> mesh_face_is_shared;
      mfem::Array<int> shared_to_interior_face;
      mfem::Array<int> inv_fp_map;

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
    // This function populates cache.face_is_shared
    // Also does some facial exploration to validate
    // Prandtl's assumptions about various face types.
    void BuildFaceLists(bool output = true)
    {
      auto *mesh = fes->GetMesh();
      auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
      auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
      int rank = Mpi::WorldRank();
      int nproc = Mpi::WorldSize();
      int nfaces_mesh = mesh->GetNumFaces();
      int nfaces_pmesh = pmesh->GetNumFaces();
      const int nfp = cache.num_face_points;
      const int neq = cache.num_equations;
      std::vector<char> is_shared(nfaces_mesh, 0);
      cache.mesh_face_is_shared.SetSize(nfaces_mesh, 0);
      std::vector<int> all_interior_faces;
      std::list<int> interior_faces; // mesh face ids of all (local) interior faces
      std::list<std::pair<int,int> > interior_face_elements; // (-,+) elid for IF
      std::list<int> shared_faces;   // mesh face ids of partition boundary faces
      std::list<std::pair<int,int> > shared_face_elements;
      std::list<int> remote_ranks;   // the remote rank for each PB face
      std::list<int> boundary_faces; // mesh face ids of domain boundary faces
      std::list<int> bc_id;          // BC id for each DB boundary face
      const int n_shared_faces = pmesh->GetNSharedFaces();
      const int ne_mesh = pmesh->GetNE();
      auto &int_faces = mesh->GetFaceIndices(mfem::FaceType::Interior);
      //      for(int face_id = 0;face_id < nfaces_mesh;face_id++){
      int num_interior_faces_mfem = int_faces.Size();
      for(int face_slot = 0;face_slot < num_interior_faces_mfem;face_slot++){
        int face_id = int_faces[face_slot];
        auto *iftr = mesh->GetInteriorFaceTransformations(face_id);
        if(iftr){
          interior_faces.push_back(face_id);
          all_interior_faces.push_back(face_id);
          interior_face_elements.push_back(std::make_pair(iftr->Elem1No,iftr->Elem2No));
          continue;
        }
        auto *sftr = pmesh->GetSharedFaceTransformationsByLocalIndex(face_id);
        if(sftr){
          cache.mesh_face_is_shared[face_id] = 1;
          is_shared[face_id] = 1;
          shared_faces.push_back(face_id);
          all_interior_faces.push_back(face_id);
          shared_face_elements.push_back(std::make_pair(sftr->Elem1No, sftr->Elem2No));
          continue;
        }
      }
      // Make sure they all detected
      MFEM_VERIFY(num_interior_faces_mfem == all_interior_faces.size(),
                  "I and MFEM disagree on interior face count.");
      int n_shared_mfem = pmesh->GetNSharedFaces();
      cache.shared_to_interior_face.SetSize(n_shared_mfem);
      for (int sfi = 0; sfi < n_shared_mfem; ++sfi)
        {
          const int face_id = pmesh->GetSharedFace(sfi);
          MFEM_VERIFY(0 <= face_id && face_id < nfaces_mesh, "bad shared face_id");
          cache.mesh_face_is_shared[face_id] = 1;
          auto slotit =                                                 \
            std::find(all_interior_faces.begin(), all_interior_faces.end(), face_id);
          if(slotit == all_interior_faces.end())
            MFEM_VERIFY(false, "Interior face id not found in interior face list");
          const int interior_face_slot = slotit - all_interior_faces.begin();
          cache.shared_to_interior_face[sfi] = interior_face_slot;
        }
      if (output){
        for(int ir = 0;ir < nproc;ir++){
          if(ir == rank){
            std::cout << "RANK: " << ir << std::endl;
            printf("nfaces_mesh = %d\n", mesh->GetNumFaces());
            printf("nfaces_pmesh = %d\n", pmesh->GetNumFaces());
            printf("restriction nf = %d\n", cache.num_interior_faces);
            std::cout << "NFaces (mesh,pmesh): (" << nfaces_mesh << ","
                      << nfaces_pmesh << ")" << std::endl;
            std::cout << "NSharedFaces: " << n_shared_faces << std::endl;
            std::cout << "Found " << interior_faces.size() << " interior faces in the mesh."
                      << std::endl;
            std::cout << "Found " << shared_faces.size() << " shared faces in the mesh."
                      << std::endl;
            std::cout << "Interior Faces: ( "; 
            std::list<int>::iterator fi = interior_faces.begin();
            while(fi != interior_faces.end()){
              int face_id = *fi++;
              std::cout << face_id << (is_shared[face_id] ? "* ": " ");
            }
            std::cout << ")" << std::endl;
            std::cout << "Shared Faces: ( ";
            fi = shared_faces.begin();
            while(fi != shared_faces.end()){
              std::cout << *fi++ << " ";
            }
            std::cout << ")" << std::endl;
            auto &int_faces = pmesh->GetFaceIndices(mfem::FaceType::Interior);
            int n_interior_mfem = int_faces.Size();
            std::cout << "MFEM says " << n_interior_mfem << " interior faces."
                      << std::endl;
            std::cout << " MFEM Interior Faces: ( ";
            for(int i = 0;i < n_interior_mfem;i++){
              int face_id = int_faces[i];
              std::cout << face_id << (is_shared[face_id] ? "* " : " ");
            }
            std::cout << ")" << std::endl;
          }
          MPI_Barrier(pmesh->GetComm());
        }
      }
      const int H = cache.restr_f->Height();
      MFEM_VERIFY(cache.num_interior_faces = num_interior_faces_mfem,
                  "Cache num_interior_faces unset");
      for (int fslot = 0; fslot < num_interior_faces_mfem; ++fslot)
        for (int side = 0; side < 2; ++side)
          for (int i = 0; i < nfp; ++i)
            for (int j = 0; j < neq; ++j)
              {
                const int idx = device_cache.iface_idx(side, i, j); // (must incorporate fslot!)
                MFEM_VERIFY(0 <= idx && idx < H, "iface_idx out of range");
              }
    }

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
      const int nfp = cache.ir_face->GetNPoints();
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
      auto &shared_face_slot = cache.shared_to_interior_face;
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
              const mfem::IntegrationPoint &ip = cache.ir_face->IntPoint(fp);
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
