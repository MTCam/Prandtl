#pragma once

namespace Prandtl
{ 
  struct DGSEMDeviceCache {
    int dim = 0;
    int num_elements = 0;
    int ndof_scalar_el = 0;
    int num_equations = 0;
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
    real_t *elWaveSpeed_d = nullptr;
    Prandtl::IdealGasModel gas;
    Prandtl::Chandrashekar::InviscidFlux iflux;
  };
}
