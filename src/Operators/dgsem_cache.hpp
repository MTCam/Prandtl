#pragma once

namespace Prandtl::DGSEM
{
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
    mfem::Array<DGSEMIntegrator *> volume_integrators;
    mfem::Array<mfem::Array<int> *> volume_element_markers;
    mfem::Array<DGSEMIntegrator *> interior_face_integrators;
    mfem::Array<BdrFaceIntegrator *> boundary_face_integrators;
    mfem::Array<mfem::Array<int> *> boundary_element_markers;

    void Output() const
    {
      std::cout << "Operator Cache:" << std::endl
                << "p = " << p << std::endl
                << "dim = " << dim << std::endl
                << "num_elements = " << num_elements << std::endl
                << "num_equations = " << num_equations << std::endl
                << "Np,Np_x,Np_y,Np_z = " << Np << "," << Np_x
                << "," << Np_y << "," << Np_z << std::endl
                << "num_face_points = " << num_face_points << std::endl
                << "num_attr = " << num_attr << std::endl
                << "ndof_scalar_el = " << ndof_scalar_el << std::endl
                << "num_interior_faces = " << num_interior_faces << std::endl;
      MFEM_VERIFY(ir, "IR is not set");
      MFEM_VERIFY(ir_face, "Face IR not set");
      MFEM_VERIFY(ir_vol, "Volume IR not set");
      MFEM_VERIFY(restr_v, "Volume Restriction not set");
      MFEM_VERIFY(restr_f, "Facial Restriction not set");
      MFEM_VERIFY(ndof_scalar_el == Np_x*Np_y*Np_z,
                  "Element dof count not equal to num quadrature points.");
      int ds_size = elem_attr.Size();
      MFEM_VERIFY(ds_size > 0, "Elem attr not set");
      ds_size = attr_marker.Size();
      MFEM_VERIFY(ds_size > 0, "Attr markers not set");
      ds_size = dnfi_marker.Size();
      MFEM_VERIFY(ds_size > 0, "dnfi markers not set");
      ds_size = elWaveSpeed.Size();
      MFEM_VERIFY(ds_size == num_elements, "Element wavespeeds missized.");
      ds_size = bndWaveSpeed.Size();
      ds_size = elJac.Size();
      MFEM_VERIFY(ds_size > 0, "Element Jacobians not set");
      ds_size = elMetric.Size();
      MFEM_VERIFY(ds_size > 0, "Element Metrics not set");
      ds_size = D.Size();
      MFEM_VERIFY(ds_size > 0, "Deriv operator not set");
      ds_size = Dhat2.Size();
      MFEM_VERIFY(ds_size > 0, "Dhat2 operator not set");
      ds_size = face_normals.Size();
      MFEM_VERIFY(ds_size == num_face_points*num_interior_faces*dim,
                  "Inapropriately sized face normals");
      ds_size = face_wt_minus.Size();
      MFEM_VERIFY(ds_size > 0, "Face weights minus not set.");
      ds_size = face_wt_plus.Size();
      MFEM_VERIFY(ds_size > 0, "Face weights plus not set.");
    }
  };
  struct DeviceCache {
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
