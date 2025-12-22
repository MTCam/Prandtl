#include "NavierStokesFlux.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{
  real_t EulerFlux::ComputeFlux(const Vector &U,
                                ElementTransformation &Tr,
                                DenseMatrix &FU) const
  {

    PointStateView S{U.GetData(), stateLayout.get()};
    // 1. Get states
    const real_t density = S.mass();              // ρ
    const Vector momentum(U.GetData()+S.momentum_eq(), dim); // ρu
    const real_t energy = S.energy();             // E, internal energy ρe
    const real_t pressure = GasModel->pressure(S);
    const real_t ke = GasModel->kinetic_energy_density(S);

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
    
    const real_t sound = GasModel->sound_speed(S);
    // fluid speed |u|
    const real_t speed = std::sqrt(2.0 * ke / density);
    // max characteristic speed = fluid speed + sound speed
    return speed + sound;
  }
  
  
  real_t EulerFlux::ComputeFluxDotN(const Vector &x,
                                    const Vector &normal,
                                    FaceElementTransformations &Tr,
                                    Vector &FUdotN) const
  {
    PointStateView S{x.GetData(), stateLayout.get()};

    // 1. Get states
    const real_t density = S.mass();                  // ρ
    const Vector momentum(x.GetData()+S.momentum_eq(), dim);  // ρu
    const real_t energy = S.energy();
    const real_t kinetic_energy = GasModel->kinetic_energy_density(S);
    const real_t pressure = GasModel->pressure(S);
    
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
    const real_t sound = GasModel->sound_speed(S);
    // fluid speed |u|
    const real_t speed = std::fabs(normal_velocity) / std::sqrt(normal*normal);
    // max characteristic speed = fluid speed + sound speed
    return speed + sound;
  }
}
