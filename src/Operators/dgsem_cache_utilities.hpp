#include "dgsem_cache.hpp"

namespace Prandtl {
  
  void OutputOperatorCache(const DGSEMOperatorCache &cache)
  {
    std::cout << "Operator Cache:" << std::endl
              << "p = " << cache.p << std::endl
              << "dim = " << cache.dim << std::endl
              << "num_elements = " << cache.num_elements << std::endl
              << "Np,Np_x,Np_y,Np_z = " << cache.Np << "," << cache.Np_x
              << "," << cache.Np_y << "," << cache.Np_z << std::endl
              << "num_face_points = " << cache.num_face_points << std::endl
              << "num_attr = " << cache.num_attr << std::endl
              << "ndof_scalar_el = " << cache.ndof_scalar_el << std::endl
              << "num_interior_faces = " << cache.num_interior_faces << std::endl;
    MFEM_VERIFY(cache.ir, "IR is not set");
    MFEM_VERIFY(cache.ir_face, "Face IR not set");
    MFEM_VERIFY(cache.ir_vol, "Volume IR not set");
    MFEM_VERIFY(cache.restr_v, "Volume Restriction not set");
    MFEM_VERIFY(cache.restr_f, "Facial Restriction not set");
    MFEM_VERIFY(cache.ndof_scalar_el == cache.Np_x*cache.Np_y*cache.Np_z,
                "Element dof count not equal to num quadrature points.");
    int ds_size = cache.elem_attr.Size();
    MFEM_VERIFY(ds_size > 0, "Elem attr not set");
    ds_size = cache.attr_marker.Size();
    MFEM_VERIFY(ds_size > 0, "Attr markers not set");
    ds_size = cache.dnfi_marker.Size();
    MFEM_VERIFY(ds_size > 0, "dnfi markers not set");
    ds_size = cache.elWaveSpeed.Size();
    MFEM_VERIFY(ds_size == cache.num_elements, "Element wavespeeds missized.");
    ds_size = cache.bndWaveSpeed.Size();
    ds_size = cache.elJac.Size();
    MFEM_VERIFY(ds_size > 0, "Element Jacobians not set");
    ds_size = cache.elMetric.Size();
    MFEM_VERIFY(ds_size > 0, "Element Metrics not set");
    ds_size = cache.D.Size();
    MFEM_VERIFY(ds_size > 0, "Deriv operator not set");
    ds_size = cache.Dhat2.Size();
    MFEM_VERIFY(ds_size > 0, "Dhat2 operator not set");
    ds_size = cache.face_normals.Size();
    MFEM_VERIFY(ds_size == cache.num_face_points*cache.num_interior_faces*cache.dim,
                "Inapropriately sized face normals");
    ds_size = cache.face_wt_minus.Size();
    ds_size = cache.face_wt_plus.Size();
    MFEM_VERIFY(ds_size > 0, "Face weights not set.");
  }

}
