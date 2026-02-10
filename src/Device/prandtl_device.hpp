#pragma once

#include "prandtl_kernels.hpp"

namespace Prandtl
{

  struct DGSEMOperatorCache
  {
    int num_elements = 0;
    int ndof_scalar_el = 0;
    int num_equations = 0;
    int num_attr = 0;
    int Np = 0;
    // Persistent device-resident aux arrays
    // Host-only handles used for gather/scatter
    const mfem::ElementRestrictionOperator *restr_v = nullptr; // for vfes (vector space)
    const mfem::ElementRestrictionOperator *restr_s = nullptr; // for fes (scalar space)
    mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
    mfem::Array<int> attr_marker;  // size nattr, 0/1
    mfem::Array<int> dnfi_marker;  // size nattr, 0/1
    mfem::Vector elJac;
    mfem::Vector elMetric;
    mfem::Vector elWaveSpeed;
    mfem::Vector D;
    mfem::Vector Dhat;
    mfem::Vector Dhat2;
    // mfem::DenseMatrix D_T;
    // mfem::DenseMatrix Dhat_T;
    // mfem::DenseMatrix Dhat2_T;
  };
}
