#pragma once
#include "Kernels.hpp"

namespace Prandtl::Device
{
  struct DGSEMCache
  {
    int num_el = 0;
    int ndof_scalar_el = 0;
    int num_eq = 0;
    int num_attr = 0;
    // Persistent device-resident aux arrays
    // Host-only handles used for gather/scatter
    mfem::ElementRestriction *restr_v = nullptr;   // for vfes (vector space)
    mfem::ElementRestriction *restr_s = nullptr;   // for fes (scalar space)
    mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
    mfem::Array<int> attr_marker;  // size nattr, 0/1
    mfem::Vector elJac;
    mfem::Vector elMetric;
    mfem::Vector elWaveSpeed;
    mfem::DenseMatrix D_T;
    mfem::DenseMatrix Dhat_T;
    mfem::DenseMatrix Dhat2_T;
  };
}
