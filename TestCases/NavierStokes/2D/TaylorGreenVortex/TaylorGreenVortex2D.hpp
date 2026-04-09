#pragma once

#include "ConditionFactory.hpp"

namespace Prandtl
{

// Taylor Green Vortex initial condition
std::function<void(const Vector&, Vector&)> TaylorGreenVortex2DIC(real_t gamma, real_t Ma)
{
    return [gamma, Ma](const Vector &x, Vector &y)
    {
        MFEM_ASSERT(x.Size() == 2, "");
        MFEM_ASSERT(y.Size() == 4, "");

        real_t den, velX, velY, energy, p, p0 = 1.0 / (gamma * Ma * Ma);

        den = 1.0;
        velX = std::sin(x(0)) * std::cos(x(1));
        velY = -std::cos(x(0)) * std::sin(x(1));
        p = p0 + 0.25 * (std::cos(2.0 * x(0)) + std::cos(2.0 * x(1)));

        energy = p / (gamma - 1.0) + 0.5 * den * (velX * velX + velY * velY);
  
        y(0) = den;
        y(1) = den * velX;
        y(2) = den * velY;
        y(3) = energy;
    };
}

// Registration helper that automatically registers these functions
struct RegisterTaylorGreenVortex2D
{
    RegisterTaylorGreenVortex2D()
    {
        // Register initial condition.
        ConditionFactory::Instance().RegisterInitialCondition2("TaylorGreenVortex2DIC", TaylorGreenVortex2DIC);
    }
};
// Global static instance to ensure registration happens at startup.
static RegisterTaylorGreenVortex2D regTaylorGreenVortex2D;

}
