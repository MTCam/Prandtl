#pragma once

namespace Prandtl {
  
  template <typename DeviceCacheT, typename GasModelT>
  MFEM_HOST_DEVICE
  real_t ApplyBoundaryConditionInviscid(const DeviceCacheT &dc,
                                        const Prandtl::BCDescriptor &bc,
                                        const real_t *state1,
                                        const real_t *nor,
                                        const real_t *aux_scalar_data,
                                        const real_t *aux_vector_data,
                                        real_t *fluxN)
  {
    const GasModelT &gas = dc.gas;
    switch (static_cast<Prandtl::BCType>(bc.type))
      {
      case Prandtl::BCType::SlipWall:
        return SlipWallInviscidFluxKernel(gas, state1, nor, fluxN);
        
      case Prandtl::BCType::SupersonicOutflow:
        return dc.iflux.ComputeFaceFlux(gas, state1, state1, nor, fluxN);

      case Prandtl::BCType::SupersonicInflow:
        {
          const real_t *bc_state = aux_vector_data + bc.primary_data_index;
          return dc.iflux(gas, state1, bc_state, nor, fluxN);
        }
        
      case Prandtl::BCType::Symmetry:
        const int neq = dc.num_equations;
        Prandtl::PointStateView S{state1};
        real_t bc_state[5];
        for(int ieq = 0;ieq < neq;ieq++){
          bc_state[5] = state1[ieq];
        }
        Prandtl::PointStateViewRW S2{bc_state};
        const int dim = dc.dim;
        real_t unorm[3];
        real_t mom[3];
        for(int idim = 0;idim < dim;dim++){
          unorm[idim] = nor[idim];
          mom[idim] = S.momentum(gas.L, idim);
        }
        Prandtl::Kernels::Normalize(dim, unorm);
        real_t nv = Prandtl::Kernels::Dot(dim, mom, unorm);
        for(int idim = 0;idim < dim;dim++){
          real_t mm = -2.0*nv*unorm[idim] + mom[idim];
          S2.set_momentum(gas.L, idim, mm);
        }
        return dc.iflux(gas, state1, bc_state, nor, fluxN);

      default:
        const neq = dc.num_equations;
        for (int eq = 0; eq < neq; ++eq) { fluxN[eq] = 0.0; }
        return 0.0;
      }
  }
}
