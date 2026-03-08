#pragma once

namespace Prandtl
{
    struct DGSEMOperatorCache {
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
      mfem::Array<int> inv_fp_map;

      // Data arrays for use on device
      mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
      mfem::Array<int> vol_attr_marker;  // size nattr, 0/1
      mfem::Array<int> domain_attr_marker;  // size nattr, 0/1
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

      // Grab the face dof from the restriction (face,point) index
      // This answers: what is the facial dof that corresponds to
      // the facial point index for a given face in the restriction?
      int MapFp(int face_slot, int fp_restr) const
      {
        return fqs_int->GetPermutedIndex(face_slot, fp_restr);
      }
      // And the inverse mapping
      int MapFpInv(int face_slot, int fp_perm) const {
        return inv_fp_map[face_slot*num_face_points + fp_perm];
      }
    };

  struct DGSEMDeviceCache {
    int p = 0;
    int dim = 0;
    int num_elements = 0;
    int ndof_scalar_el = 0;
    int num_equations = 0;
    int num_face_points = 0;
    int num_interior_faces = 0;
    int Np_x = 0;
    int Np_y = 0;
    int Np_z = 0;
    int Np = 0;
    int num_attr = 0;
    const int *elem_attr_d = nullptr;    // size ne, values are 1-based attributes
    const int *attr_marker_d = nullptr;  // size nattr, 0/1
    const real_t *elJac_d = nullptr;
    const real_t *elMetric_d = nullptr;
    const real_t *D_d = nullptr;
    const real_t *Dhat_d = nullptr;
    const real_t *Dhat2_d = nullptr;
    const real_t *nor_d = nullptr;
    const real_t *fw_minus_d = nullptr;
    const real_t *fw_plus_d = nullptr;
    real_t *elWaveSpeed_d = nullptr;
    real_t *ifWaveSpeed_d = nullptr;
    real_t *bndWaveSpeed_d = nullptr;
    Prandtl::IdealGasModel gas;
    Prandtl::ChandrashekarFlux::InviscidFlux iflux;


    MFEM_HOST_DEVICE inline int iface_idx(int side, int fp, int eq) const
    {
      return ((side*num_equations + eq)*num_face_points + fp);
    }
    MFEM_HOST_DEVICE inline int iface_size() const
    {
      return 2*num_equations*num_face_points;
    }
  };
}
