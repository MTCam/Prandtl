#pragma once

#include "bc_cache_utilities.hpp"
#include "Flow.hpp"

namespace Prandtl {

  namespace BC {


    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE inline
    void ComputeBdrFaceGradFlux(const DeviceCacheT &dc,
                                const Prandtl::BCDescriptor &bc,
                                const real_t *state1,
                                real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const int neq = dc.num_equations;
      const int dim = dc.dim;
      
      for (int q = 0; q < neq; ++q)
        {
          fluxN[q] = real_t(0);
        }

      switch (static_cast<Prandtl::BCType>(bc.type))
        {
        case Prandtl::BCType::NoSlipAdiab:
          {
            const real_t *vector_data = dc.bc_vector_d;
            const real_t *Vwall = vector_data + bc.data_index;
            const real_t *wallHeat = vector_data + bc.data_index + dim;

            // Note this must be entropy state
            Prandtl::PointStateView S{state1};
            Prandtl::PointStateViewRW F{fluxN};

            const real_t v = -gas.energy(S);

            // Build the same provisional "boundary" state as legacy, then subtract state1.
            F.set_mass(gas.L, gas.mass(S));
            for(int idim = 0;idim < dim;idim++){
              F.set_momentum(gas.L, idim, Vwall[idim] * v);
            }
            F.set_energy(gas.L, -v);
            for (int q = 0; q < neq; ++q)
              {
                fluxN[q] -= state1[q];
              }
            return;
          }

        default:
          {
            // Conservative placeholder for unsupported BCs.
            for (int q = 0; q < neq; ++q)
              {
                fluxN[q] = real_t(0);
              }
            return;
          }
        }
    }


    template<typename GasModelT>
    MFEM_HOST_DEVICE
    real_t SlipWallInviscidFluxKernel(const GasModelT &gasModel, const real_t *state1,
                                      const real_t *nor, real_t *fluxN)
    {
      
      real_t unit_nor[Prandtl::MAXDIM];
      real_t state2[Prandtl::MAXEQ];
      const int dim = gasModel.L.dim;
      const int neq = gasModel.L.nequations();
      for(int idim = 0;idim < dim;idim++)
        unit_nor[idim] = nor[idim];
      for(int ieq = 0;ieq < neq;ieq++){
        state2[ieq] = state1[ieq];
        fluxN[ieq] = 0.0;
      }
      Prandtl::Kernels::Normalize(dim, unit_nor);
      Prandtl::PointStateViewRW S{state2};
      Prandtl::Flow::RotateState(gasModel.L, unit_nor, S);
      const real_t p_star = Prandtl::Flow::slipwall_pstar(S, gasModel);
      const real_t v = gasModel.velocity(S, 0); // the "x" component is v*n
      const real_t c = gasModel.sound_speed(S);
      const int mom_eq = gasModel.L.eq_mom0;
      for(int idim = 0;idim < dim;idim++)
        fluxN[mom_eq+idim] = p_star * nor[idim];
      return std::abs(v) + c;
    }

    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE
    real_t ApplyBoundaryConditionInviscid(const DeviceCacheT &dc,
                                          const Prandtl::BCDescriptor &bc,
                                          const real_t *state1,
                                          const real_t *nor,
                                          real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const real_t *scalar_data = dc.bc_scalar_d;
      const real_t *vector_data = dc.bc_vector_d;
      switch (static_cast<Prandtl::BCType>(bc.type))
        {
        case Prandtl::BCType::SlipWall:
          return SlipWallInviscidFluxKernel(gas, state1, nor, fluxN);
          
        case Prandtl::BCType::SupersonicOutflow:
          return dc.iflux.ComputeFaceFlux(gas, state1, state1, nor, fluxN);
          
        case Prandtl::BCType::SupersonicInflow:
          {
            const real_t *bc_state = vector_data + bc.data_index;
            return dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
          }
          
        case Prandtl::BCType::Symmetry:
          {
            const int neq = dc.num_equations;
            Prandtl::PointStateView S{state1};
            real_t bc_state[Prandtl::MAXEQ];
            for(int ieq = 0;ieq < neq;ieq++){
              bc_state[ieq] = state1[ieq];
            }
            Prandtl::PointStateViewRW S2{bc_state};
            const int dim = dc.dim;
            real_t unorm[Prandtl::MAXDIM];
            real_t mom[Prandtl::MAXDIM];
            for(int idim = 0;idim < dim;idim++){
              unorm[idim] = nor[idim];
              mom[idim] = S.momentum(gas.L, idim);
            }
            Prandtl::Kernels::Normalize(dim, unorm);
            real_t nv = Prandtl::Kernels::Dot(dim, mom, unorm);
            for(int idim = 0;idim < dim;idim++){
              real_t mm = -2.0*nv*unorm[idim] + mom[idim];
              S2.set_momentum(gas.L, idim, mm);
            }
            return dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
          }
        default:
          {
            const int neq = dc.num_equations;
            for (int eq = 0; eq < neq; ++eq) { fluxN[eq] = 0.0; }
            return 0.0;
          }
        }
      return 0.0;
    }
  }
}
