#include "NavierStokesFlux.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{

  void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy,
                                            const Vector &dqdz, DenseMatrix &flux) const
  {
    PointStateView S{state.GetData()};
    real_t mu = gasModel->viscosity(S);
    real_t kappa = gasModel->thermal_conductivity(S);
    real_t mu_bulk_loc = gasModel->bulk_viscosity(S);    
    
    const real_t &drdx = dqdx(0);
    const real_t &dudx = dqdx(1);
    const real_t &dvdx = dqdx(2);
    const real_t &dwdx = dqdx(3);
    const real_t &dpdx = dqdx(4);
    
    const real_t &drdy = dqdy(0);
    const real_t &dudy = dqdy(1);
    const real_t &dvdy = dqdy(2);
    const real_t &dwdy = dqdy(3);
    const real_t &dpdy = dqdy(4);

    const real_t &drdz = dqdz(0);
    const real_t &dudz = dqdz(1);
    const real_t &dvdz = dqdz(2);
    const real_t &dwdz = dqdz(3);
    const real_t &dpdz = dqdz(4);

    const real_t grad_rho[3] = {drdx, drdy, drdz};
    const real_t grad_p[3] = {dpdx, dpdy, dpdz};
    real_t grad_t[3] = {0.0, 0.0, 0.0};

    real_t vx = gasModel->velocity(S, 0);
    real_t vy = gasModel->velocity(S, 1);
    real_t vz = gasModel->velocity(S, 2);

    gasModel->grad_temperature(S, grad_rho, grad_p, grad_t);

    real_t div = dudx + dvdy + dwdz;

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk_loc * div);
    flux(2, 0) = mu * (dudy + dvdx);
    flux(3, 0) = mu * (dudz + dwdx);
    flux(4, 0) = vx * flux(1, 0) + vy * flux(2, 0) + vz * flux(3, 0) + kappa * grad_t[0];

    flux(1, 1) = mu * (dvdx + dudy);
    flux(2, 1) = mu * (2.0 * dvdy - mu_bulk_loc * div);
    flux(3, 1) = mu * (dvdz + dwdy);
    flux(4, 1) = vx * flux(1, 1) + vy * flux(2, 1) + vz * flux(3, 1) + kappa * grad_t[1];

    flux(1, 2) = mu * (dwdx + dudz);
    flux(2, 2) = mu * (dwdy + dvdz);
    flux(3, 2) = mu * (2.0 * dwdz - mu_bulk_loc * div);
    flux(4, 2) = vx * flux(1, 2) + vy * flux(2, 2) + vz * flux(3, 2) + kappa * grad_t[2]; 
  }
  
  void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx,
                                            const Vector &dqdy, DenseMatrix &flux) const
  {
    PointStateView S{state.GetData()};
    real_t kappa = gasModel->thermal_conductivity(S);
    real_t mu = gasModel->viscosity(S);
    real_t mu_bulk_loc = gasModel->bulk_viscosity(S);    
    real_t press = gasModel->pressure(S);
    
    const real_t &drdx = dqdx(0);
    const real_t &dudx = dqdx(1);
    const real_t &dvdx = dqdx(2);
    const real_t &dpdx = dqdx(3);
    
    const real_t &drdy = dqdy(0);
    const real_t &dudy = dqdy(1);
    const real_t &dvdy = dqdy(2);
    const real_t &dpdy = dqdy(3);

    const real_t grad_rho[2] = {drdx, drdy};
    const real_t grad_p[2] = {dpdx, dpdy};
    real_t grad_t[2] = {0.0, 0.0};
    real_t vx = gasModel->velocity(S, 0);
    real_t vy = gasModel->velocity(S, 1);
    real_t rho = gasModel->density(S);

    gasModel->grad_temperature(S, grad_rho, grad_p, grad_t);
    real_t div = dudx + dvdy;

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk_loc * div);
    flux(2, 0) = mu * (dudy + dvdx);
    flux(3, 0) = vx * flux(1, 0) + vy * flux(2, 0) + kappa * grad_t[0];

    flux(1, 1) = mu * (dvdx + dudy);
    flux(2, 1) = mu * (2.0 * dvdy - mu_bulk_loc * div);
    flux(3, 1) = vx * flux(1, 1) + vy * flux(2, 1) + kappa * grad_t[1];
}

  void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, DenseMatrix &flux) const
  {
    PointStateView S{state.GetData()};
    real_t kappa = gasModel->thermal_conductivity(S);
    real_t mu = gasModel->viscosity(S);
    real_t mu_bulk_loc = gasModel->bulk_viscosity(S);    
    
    const real_t &drdx = dqdx(0);
    const real_t &dudx = dqdx(1);
    const real_t &dpdx = dqdx(2);
    
    const real_t grad_rho[1] = {drdx};
    const real_t grad_p[1] = {dpdx};
    real_t grad_t[1] = {0.0};
    real_t vx = gasModel->velocity(S, 0);
    gasModel->grad_temperature(S, grad_rho, grad_p, grad_t);
    real_t div = dudx;
    
    flux(1, 0) = mu * (2.0 * dudx - mu_bulk_loc * div);
    flux(2, 0) = vx * flux(1, 0) + kappa * grad_t[0];
  }
  
  real_t NavierStokesFlux::ComputeFlux(const Vector &U,
                                       ElementTransformation &Tr,
                                       DenseMatrix &FU) const
  {
    
    PointStateView S{U.GetData()};
    
    // 1. Get states
    const real_t density = gasModel->density(S);
    const Vector momentum(U.GetData()+gasModel->L.eq_mom0, dim);
    const real_t energy = gasModel->energy(S);
    const real_t pressure = gasModel->pressure(S);
    const real_t ke = gasModel->kinetic_energy_density(S);
    
    // Check whether the solution is physical only in debug mode
    MFEM_ASSERT(density >= 0, "Negative Density");
    MFEM_ASSERT(pressure >= 0, "Negative Pressure");
    MFEM_ASSERT(energy >= 0, "Negative Energy");
    
    // 2. Compute Flux
    for (int d = 0; d < dim; d++)
      {
        FU(0, d) = momentum(d);  // ρu
        for (int i = 0; i < dim; i++)
          {
            // ρuuᵀ
            FU(1 + i, d) = momentum(i) * momentum(d) / density;
          }
        // (ρuuᵀ) + p
        FU(1 + d, d) += pressure;
      }
    // enthalpy H = e + p/ρ = (E + p)/ρ
    const real_t H = (energy + pressure) / density;
    for (int d = 0; d < dim; d++)
      {
        // u(E+p) = ρu*(E + p)/ρ = ρu*H
        FU(1 + dim, d) = momentum(d) * H;
      }
    
    // 3. Compute maximum characteristic speed
    
    const real_t sound = gasModel->sound_speed(S);
    // fluid speed |u|
    const real_t speed = std::sqrt(2.0 * ke / density);
    // max characteristic speed = fluid speed + sound speed
    return speed + sound;
  }
  
  
  real_t NavierStokesFlux::ComputeFluxDotN(const Vector &x,
                                           const Vector &normal,
                                           FaceElementTransformations &Tr,
                                           Vector &FUdotN) const
  {
    PointStateView S{x.GetData()};

    // 1. Get states
    const real_t density = gasModel->density(S);
    const Vector momentum(x.GetData()+gasModel->L.eq_mom0, dim);  // ρu
    const real_t energy = gasModel->energy(S);
    const real_t kinetic_energy = gasModel->kinetic_energy_density(S);
    const real_t pressure = gasModel->pressure(S);
    
    // Check whether the solution is physical only in debug mode
    MFEM_ASSERT(density >= 0, "Negative Density");
    MFEM_ASSERT(pressure >= 0, "Negative Pressure");
    MFEM_ASSERT(energy >= 0, "Negative Energy");
    
    // 2. Compute normal flux
    FUdotN(0) = momentum * normal;  // ρu⋅n
    // u⋅n
    const real_t normal_velocity = FUdotN(0) / density;
    for (int d = 0; d < dim; d++)
      {
        // (ρuuᵀ + pI)n = ρu*(u⋅n) + pn
        FUdotN(1 + d) = normal_velocity * momentum(d) + pressure * normal(d);
      }
    // (u⋅n)(E + p)
    FUdotN(1 + dim) = normal_velocity * (energy + pressure);
    
    // 3. Compute maximum characteristic speed
    const real_t sound = gasModel->sound_speed(S);
    // fluid speed |u|
    const real_t speed = std::fabs(normal_velocity) / std::sqrt(normal*normal);
    // max characteristic speed = fluid speed + sound speed
    return speed + sound;
  }

}
