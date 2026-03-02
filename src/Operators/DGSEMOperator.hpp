#pragma once

#include "DGSEMIntegrator.hpp"
#include "DGSEMNonlinearForm.hpp"
#include "BdrFaceIntegrator.hpp"
#include "ModalBasis.hpp"
#include "Indicator.hpp"
#include "BasicOperations.hpp"
#include "GasModel.hpp"
#include "prandtl_device.hpp"
#include "dgsem_cache.hpp"

namespace Prandtl
{
  
  using namespace mfem;
  
  class DGSEMOperator : public TimeDependentOperator
  {
  private:
    std::shared_ptr<ParFiniteElementSpace> vfes;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::shared_ptr<ParMesh> pmesh;
    std::shared_ptr<ParGridFunction> eta;
    std::shared_ptr<ParGridFunction> alpha;
    std::vector<std::shared_ptr<ParGridFunction> > grad_u;
    std::shared_ptr<ParGridFunction> r_gf;
    std::unique_ptr<DGSEMIntegrator> integrator;
    std::unique_ptr<Indicator> indicator;
    const IdealGasModel gasModel;
    std::unique_ptr<DGSEMNonlinearForm> nonlinearForm; 
    Prandtl::DGSEM::OperatorCache cache;
    Prandtl::DGSEM::DeviceCache device_cache;
    mutable Array<int> vdof_indices;
    mutable Vector el_vdofs, grad_vdofs;
    
    const int num_equations, dim, order, num_elements;
    const int num_dofs_scalar;
    const int Ndofs;
    
    mutable Vector global_entropy;
    
#ifdef AXISYMMETRIC
    mutable long long calls_accum = 0, highOrder_shape_accum = 0, lowOrder_ray2_accum = 0, lowOrder_ray1_accum = 0, lowOrder_copy_accum = 0;
    mutable Vector U;
    Array<int> axis_marker;
    Array<int> axis_idx;
    bool low_order_axis = false;
    real_t rho_floor_abs = 1e-12;
    real_t p_floor_abs = 1e-12;
    bool highOrder_near_axis_toggle = false;
    real_t highOrder_near_axis_fraction = 0.5; // 0.0 -> 100% highOrder, 1.0 -> 100% lowOrder
#endif
    
    const real_t sharpness_fac = 9.21024;
    const real_t modalThreshold;
    const real_t alpha_min;
    const real_t alpha_max;
    
    mutable real_t max_char_speed;
    
    std::vector<BdrFaceIntegrator*> bfnfi;
    std::vector<Array<int>> bdr_marker;
    mutable Array<int> ind_indx;
    mutable Vector ind_dof;
    mutable real_t alpha_dof;
    const mfem::ElementRestrictionOperator *restr_v = nullptr; // for vfes (vector space)
    const mfem::ElementRestrictionOperator *restr_s = nullptr; // for fes (scalar space)
    
    void ComputeGlobalEntropyVector(const Vector &u, Vector &global_entropy) const;
    void ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx) const;
    void ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy) const;
    void ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const;
    void ComputeBlendingCoefficient(const Vector &u) const;
    // void AssembleDeviceCache();
    
#ifdef AXISYMMETRIC
    void BuildAxisIndexFromMarker();
    void ZeroAxisRadialMom(Vector &v) const;
    void ResetAxisReconStats() const
    {
      calls_accum = highOrder_shape_accum = lowOrder_ray2_accum = lowOrder_ray1_accum = lowOrder_copy_accum = 0;
    }
    
    struct AxisReconStats {
      long long calls, highOrder_shape, lowOrder_ray2, lowOrder_ray1, lowOrder_copy;
    };
#endif
    
  public:
    DGSEMOperator(std::shared_ptr<ParFiniteElementSpace> vfes,
                  std::shared_ptr<ParFiniteElementSpace> fes0,
                  std::shared_ptr<ParMesh> pmesh,
                  std::shared_ptr<ParGridFunction> eta,
                  std::shared_ptr<ParGridFunction> alpha,
                  std::vector<std::shared_ptr<ParGridFunction> > &grad_u_,
                  std::unique_ptr<DGSEMIntegrator> integrator,
                  std::unique_ptr<Indicator> indicator,
                  const IdealGasModel &gasModel_,
                  std::shared_ptr<ParGridFunction> r_gf = nullptr,
                  const real_t alpha_max = 0.5, const real_t alpha_min = 0.001);
    
    ~DGSEMOperator();
    
    // void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker);
    
    void Mult(const Vector &u, Vector &dudt) const override;
    inline real_t GetMaxCharSpeed()
    {
      return max_char_speed;
    }
    
    inline real_t& GetTimeRef()
    {
      return t;
    }
    
#ifdef AXISYMMETRIC
    void RecoverStateFromWeighted(const Vector &rU, Vector &U) const;
    long long GetTotalFallbacks1st(bool global = true) const;
    long long GetTotalFallbacks0th(bool global = true) const;
    AxisReconStats GetAxisReconStats(bool global = true) const;
    void SetAxisBoundaryMarker(const Array<int>& marker) 
    { 
      axis_marker = marker;
      BuildAxisIndexFromMarker();
    }
    void SetLowOrderAxis(bool enable) 
    {
      low_order_axis = enable;
    }
    bool GetLowOrderAxis() const
    {
      return low_order_axis;
    }
    void SetAxisFloorsFromFreestream(real_t rho_inf, real_t p_inf, real_t rho_fac = 1e-8, real_t p_fac = 1e-8)
    {
      rho_floor_abs = std::max(rho_floor_abs, rho_fac * rho_inf);
      p_floor_abs = std::max(p_floor_abs, p_fac * p_inf);
    }
#endif
    void AddDomainIntegrator(DGSEMIntegrator *nlfi)
    {
      cache.volume_integrators.Append(nlfi);
      cache.volume_element_markers.Append(NULL);
    }
    void AddInteriorFaceIntegrator(DGSEMIntegrator *nlfi)
    {
      cache.interior_face_integrators.Append(nlfi);
    }
    
    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker)
    {
      cache.boundary_face_integrators.Append(bfi);
      cache.boundary_element_markers.Append(&bdr_marker);
      nonlinearForm->AddBdrFaceIntegrator(bfi, bdr_marker);
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
    void AssembleGeometricTerms();
    void AssembleElementVolumeGeometricTerms(mfem::ElementTransformation &);
    void AssembleInteriorFaceGeomCache();
    void CreateOperatorCache();
    void OperatorCacheToDeviceCache();
    void BuildFaceLists(bool output = true)
    {
      //auto *mesh = fes->GetMesh();
      // auto *pfes = vfes; // dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
      //auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
      int rank = Mpi::WorldRank();
      int nproc = Mpi::WorldSize();
      // int nfaces_mesh = pmesh->GetNumFaces();
      int nfaces_in_mesh = pmesh->GetNumFaces();
      const int nfp = cache.num_face_points;
      const int neq = cache.num_equations;
      std::vector<char> is_shared(nfaces_in_mesh, 0);
      cache.mesh_face_is_shared.SetSize(nfaces_in_mesh, 0);
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
      auto &int_faces = pmesh->GetFaceIndices(mfem::FaceType::Interior);
      //      for(int face_id = 0;face_id < nfaces_mesh;face_id++){
      int num_interior_faces_mfem = int_faces.Size();
      for(int face_slot = 0;face_slot < num_interior_faces_mfem;face_slot++){
        int face_id = int_faces[face_slot];
        auto *iftr = pmesh->GetInteriorFaceTransformations(face_id);
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
          MFEM_VERIFY(0 <= face_id && face_id < nfaces_in_mesh, "bad shared face_id");
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
            printf("nfaces_in_mesh = %d\n", nfaces_in_mesh);
            printf("restriction nf = %d\n", cache.num_interior_faces);
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
      for (int fslot = 0; fslot < num_interior_faces_mfem; ++fslot){
        for (int side = 0; side < 2; ++side){
          for (int i = 0; i < nfp; ++i){
            for (int j = 0; j < neq; ++j)
              {
                const int idx = device_cache.iface_idx(side, i, j); // (must incorporate fslot!)
                MFEM_VERIFY(0 <= idx && idx < H, "iface_idx out of range");
              } // eq
          } // fp
        } // side
      } // slot
    } // BuildFaceLists
  };  
}
