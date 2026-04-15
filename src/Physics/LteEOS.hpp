#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"
#include "../BasicOperations/BasicOperations.hpp"

namespace Prandtl
{

// ============================================================================
// EOS: Gas-mixture in local thermodynamic equilibrium
// ============================================================================
  struct LTEGasEOS
  {
    // ---- helpers on conservative state --------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t R_gas(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return bilinear_interpolate(L.R_eq_idx, phys, L, S);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t density(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
        return S.mass(L); // this is "rho" (mass density)
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t rhoE(const PhysicsConstants &phys, const StateLayout &L,
                       const StateView &S) const
    {
        return S.energy(L);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t momentum_sq(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
        const int dim = L.dim;   // uses state layout
        real_t m2 = 0;
        for (int d = 0; d < dim; ++d)
        {
          const real_t m = S.momentum(L,d);
          m2 += m * m;
        }
        return m2;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                         const StateView &S) const
    {
      // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
      const real_t rho  = density(phys, L, S);
      const real_t m2   = momentum_sq(phys, L, S);
      return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                          const StateView &S) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return rhoE(phys, L, S) - kinetic_energy_density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_from_pressure(const PhysicsConstants &phys, const StateLayout &L,
                                                const StateView &S, real_t pressure_target) const
    {
      real_t U[L.eq_energy + 1]; // CL NOTE : Need a MACRO for size
      PointStateViewRW S_dummy(U);
      S_dummy.set_mass(L, S.mass(L));
      for(int idim = 0; idim < L.dim; idim++)
      {
        S_dummy.set_momentum(L, idim, S.momentum(L, idim));
      }
      S_dummy.set_energy(L, S.energy(L));

      // Secant Method to find internal energy that matches a target pressure
      real_t tol   = 1e-12;
      real_t denom = 0.0;
      real_t ie_old = internal_energy_density(phys, L, S);
      real_t ie_new = ie_old*1.01 + 1e-12;

      real_t ke = kinetic_energy_density(phys, L, S);

      real_t f_old, f_new, ie_update;

      for(int iter = 0; iter < 100; iter++)
      {
        S_dummy.set_energy(L, ie_old+ke);
        f_old = bilinear_interpolate(L.P_idx, phys, L, S_dummy) - pressure_target;

        S_dummy.set_energy(L, ie_new+ke);
        f_new = bilinear_interpolate(L.P_idx, phys, L, S_dummy) - pressure_target;

        denom = f_new - f_old;

        if( std::abs(f_new) < tol || std::abs(denom) < 1e-12)
        {
          break;
        }

        ie_update = ie_new - f_new * (ie_new - ie_old) / denom;

        ie_old = ie_new;
        ie_new = ie_update;
      }

      return ie_new;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                           const StateView &S) const
    {
        // e = (rho*e) / rho
      const real_t rho  = density(phys, L, S);
      const real_t rhoe = internal_energy_density(phys, L, S);
      return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t pressure(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S) const
    {
      return bilinear_interpolate(L.P_idx, phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t gamma(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return bilinear_interpolate(L.gamma_eq_idx, phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t temperature(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      return bilinear_interpolate(L.T_idx, phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const PhysicsConstants &phys, const StateLayout  &L,
                                 const StateView &S, const real_t *grad_rho,
                                 const real_t *grad_p, real_t *grad_t) const
    {
      for(int i = 0; i < L.dim; i++){
        grad_t[i] = grad_p[i]; // CL NOTE : we store T_xi in contiguous gradient array for LTE (W=[rho, u, v, w , T])
      }     
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      return bilinear_interpolate(L.c_idx, phys, L, S);
    }
    
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t cv(const PhysicsConstants &phys, const StateLayout &L,
                     const StateView &S) const
    {
      return bilinear_interpolate(L.cv_idx, phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t cp(const PhysicsConstants &phys, const StateLayout &L,
                     const StateView &S) const
    {
        return cv(phys, L, S) * gamma(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t entropy(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
      return bilinear_interpolate(L.s_idx, phys, L, S);
      return 0.0;
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const PhysicsConstants &phys, const StateLayout &L,
                              const InStateView &S, OutStateView &E) const
    {
      const real_t p    = pressure(phys, L, S);
      const real_t rho  = S.mass(L);
      const real_t T    = temperature(phys, L, S);
      const real_t s    = entropy(phys, L, S);
      const real_t e    = specific_internal_energy(phys, L, S);
      const real_t v2o2 = kinetic_energy_density(phys, L, S) / rho;
      const real_t beta = 1/T;

      const real_t ent_1 = (e + p/rho - v2o2)*beta - s;

      E.set_mass(L, ent_1);
      int dim = L.dim;
      int num_scalars = L.num_scalars;
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(L, idim, beta * S.velocity(L, idim));
      }
      E.set_energy(L, -beta);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const PhysicsConstants &phys, const StateLayout &L,
                                          const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim) const
    {

      const real_t T    = temperature(phys, L, S);

      dPrim.set_mass(L, 0.0); // CL NOTE : won't be using density gradient (if W = [rho, u, v, w , T])
      int dim = L.dim;
      for(int i=0; i < dim; i++)
      {
        dPrim.set_momentum(L, i, dE.momentum(L, i)*T + T*S.velocity(L, i)*dE.energy(L));
      }
      dPrim.set_energy(L, T*T*dE.energy(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                     const InStateView &Se, OutStateView &Sc) const
    {
      std::cerr<< " CL ALERT : Not functional in LTE yet "<<std::endl;
    }

    template<typename InStateView, typename OutStateView>
    inline void primitive_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                       const InStateView &prim, OutStateView &cons) const
    {
      const real_t rho = prim.mass(L);
      const int dim    = L.dim;
      real_t v2        = 0.0;

      cons.set_mass(L, rho);
      for(int d = 0; d < dim; d++)
      {
        cons.set_momentum(L, d, rho*prim.velocity(L, d));
        v2 += prim.velocity(L,d)*prim.velocity(L,d);
      }
      real_t rhoe = internal_energy_from_pressure(phys, L, cons, prim.pressure(L));
      cons.set_energy(L, rhoe + 0.5 * rho * v2);
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const PhysicsConstants &phys, const StateLayout &L,
                         const StateView &S, real_t u[3]) const
    {
      const int dim = L.dim;
      for (int d = 0; d < dim; ++d)
        {
          u[d] = S.velocity(L, d);
        }
      for (int d = dim; d < 3; ++d)
        {
          u[d] = real_t(0);
        }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t bilinear_interpolate(int property_idx, const PhysicsConstants &phys,
                                       const StateLayout &L, const StateView &S) const
    {
      /*
      * Q01--------Q11
      *  |          |
      *  |          |
      *  |          |
      * Q00--------Q10
      */

      // Point rho and rhoe values
      real_t rho  = density(phys, L, S);
      real_t rhoe = internal_energy_density(phys, L, S);      

      // CL NOTE : Find a way to make a good first guess of l_x and l_y (because we are not storing it in cache)

      // Get the lower and upper x and y indices of the cell
      int l_y = hunt(phys.rhoe_grid, L.ny, rhoe, 0);
      int l_x = hunt(phys.rho_grid, L.nx, rho, 0);
      int u_x = l_x + 1  , u_y = l_y + 1;

      if(l_x < 0 || u_x >= L.nx || l_y < 0 || u_y >= L.ny)
      {
        std::cout << " CL ALERT : Out of bounds in LTE table lookup! "<<std::endl;
        std::cout << "l_x : " << l_x << "l_y : " << l_y << std::endl;
        std::cout << "rho : " << phys.rho_grid[0] << " < " << rho << " < " << phys.rho_grid[L.nx-1] << std::endl;
        std::cout << "rhoe : " << phys.rhoe_grid[0] << " < " << rhoe << " < " << phys.rhoe_grid[L.ny-1] << std::endl;
        std::exit(1);
      }

      // Get the lower and upper x and y coordinates of the cell
      real_t rho_l  = phys.rho_grid[l_x] , rho_u = phys.rho_grid[u_x];
      real_t rhoe_l = phys.rhoe_grid[l_y], rhoe_u = phys.rhoe_grid[u_y];

      // Get the corner property values
      real_t Q00 = phys.lte_table[L.lte_property_index(property_idx, l_x, l_y)];
      real_t Q01 = phys.lte_table[L.lte_property_index(property_idx, l_x, u_y)];
      real_t Q10 = phys.lte_table[L.lte_property_index(property_idx, u_x, l_y)];
      real_t Q11 = phys.lte_table[L.lte_property_index(property_idx, u_x, u_y)];


      real_t wx = (rho  - rho_l)  / (rho_u - rho_l);
      real_t wy = (rhoe - rhoe_l) / (rhoe_u - rhoe_l);


      // Clamp to [0, 1]
      wx = std::max(real_t(0), std::min(real_t(1), wx));
      wy = std::max(real_t(0), std::min(real_t(1), wy));


      return Q00 * ((1 - wx) * (1 - wy)) + Q01 * ((1 - wx) * wy) +
            Q10 * (wx * (1 - wy)) + Q11 * (wx * wy);
    }
  };
}
