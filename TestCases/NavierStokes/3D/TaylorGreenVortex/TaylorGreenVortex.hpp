#pragma once

#include "ConditionFactory.hpp"

namespace Prandtl
{

// Taylor Green Vortex initial condition
std::function<void(const Vector&, Vector&)> TaylorGreenVortexIC(real_t gamma, real_t Ma)
{
    return [gamma, Ma](const Vector &x, Vector &y)
    {
        MFEM_ASSERT(x.Size() == 3, "");

        real_t den, velX, velY, velZ, energy, p, p0 = 1.0 / (gamma * Ma * Ma);

        den = 1.0;
        velX = std::sin(x(0)) * std::cos(x(1)) * std::cos(x(2));
        velY = -std::cos(x(0)) * std::sin(x(1)) * std::cos(x(2));
        velZ = 0.0;
        p = p0 + 1.0 / 16.0 * (std::cos(2.0 * x(0)) + std::cos(2.0 * x(1))) * std::cos(2.0 * x(2) + 2);

        energy = p / (gamma - 1.0) + 0.5 * den * (velX * velX + velY * velY + velZ * velZ);
  
        y(0) = den;
        y(1) = den * velX;
        y(2) = den * velY;
        y(3) = den * velZ;
        y(4) = energy;
    };
}

// Registration helper that automatically registers these functions
struct RegisterTaylorGreenVortex
{
    RegisterTaylorGreenVortex()
    {
        // Register initial condition.
        ConditionFactory::Instance().RegisterInitialCondition2("TaylorGreenVortexIC", TaylorGreenVortexIC);
    }
};
// Global static instance to ensure registration happens at startup.
static RegisterTaylorGreenVortex regTaylorGreenVortex;

}
