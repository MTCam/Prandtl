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
    void BuildFaceLists(bool output = false)
    {
      auto *mesh = fes->GetMesh();
      auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
      auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
      int rank = Mpi::WorldRank();
      int nproc = Mpi::WorldSize();
      int nfaces_mesh = mesh->GetNumFaces();
      int nfaces_pmesh = pmesh->GetNumFaces();
      const int nfp = cache->num_face_points;
      const int neq = cache->num_equations;
      std::vector<char> is_shared(nfaces_mesh, 0);
      cache->mesh_face_is_shared.SetSize(nfaces_mesh, 0);
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
          cache->mesh_face_is_shared[face_id] = 1;
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
      cache->shared_to_interior_face.SetSize(n_shared_mfem);
      for (int sfi = 0; sfi < n_shared_mfem; ++sfi)
        {
          const int face_id = pmesh->GetSharedFace(sfi);
          MFEM_VERIFY(0 <= face_id && face_id < nfaces_mesh, "bad shared face_id");
          cache->mesh_face_is_shared[face_id] = 1;
          auto slotit =                                                 \
            std::find(all_interior_faces.begin(), all_interior_faces.end(), face_id);
          if(slotit == all_interior_faces.end())
            MFEM_VERIFY(false, "Interior face id not found in interior face list");
          const int interior_face_slot = slotit - all_interior_faces.begin();
          cache->shared_to_interior_face[sfi] = interior_face_slot;
        }
      if (output){
        for(int ir = 0;ir < nproc;ir++){
          if(ir == rank){
            std::cout << "RANK: " << ir << std::endl;
            printf("nfaces_mesh = %d\n", mesh->GetNumFaces());
            printf("nfaces_pmesh = %d\n", pmesh->GetNumFaces());
            printf("restriction nf = %d\n", cache->num_interior_faces);
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
      const int H = cache->restr_f->Height();
      MFEM_VERIFY(cache->num_interior_faces = num_interior_faces_mfem,
                  "Cache num_interior_faces unset");
      for (int fslot = 0; fslot < num_interior_faces_mfem; ++fslot)
        for (int side = 0; side < 2; ++side)
          for (int i = 0; i < nfp; ++i)
            for (int j = 0; j < neq; ++j)
              {
                const int idx = device_cache->iface_idx(side, i, j); // (must incorporate fslot!)
                MFEM_VERIFY(0 <= idx && idx < H, "iface_idx out of range");
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
