#pragma once

#include "ConditionFactory.hpp"

namespace Prandtl
{

// Isentropic Vortex initial condition
std::function<void(const Vector&, Vector&)> LTEVortexIC(real_t radius,
                                                        real_t vel_inf,
                                                        real_t beta,
                                                        real_t rho_inf,
                                                        real_t temp_inf)
{
    return [=](const Vector &x, Vector &y)
    {
        MFEM_ASSERT(x.Size() == 2, "");

        const real_t xc = 0.0;
        const real_t yc = 0.0;

        // Using CPG constants only to shape the initial field.
        const real_t gamma = 1.4;
        const real_t R_gas = 287.05;
        const real_t gm1   = gamma - 1.0;
        const real_t cp    = gamma * R_gas / gm1;

        // using the perfect-gas relations
        const real_t pres_inf = rho_inf * R_gas * temp_inf;

        real_t dx = x(0) - xc;
        real_t dy = x(1) - yc;

        real_t r2rad = (dx*dx + dy*dy) / (radius * radius);

        const real_t exp_half = std::exp(-0.5 * r2rad);
        const real_t exp_full = std::exp(-r2rad);

        // Vortex velocity field
        const real_t velX = vel_inf * (1.0 - beta * dy / radius * exp_half);
        const real_t velY = vel_inf * (      beta * dx / radius * exp_half);
        const real_t vel2 = velX * velX + velY * velY;

        // Gaussian temperature perturbation
        const real_t temp =
            temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) / cp * exp_full;

        // Safety clamp so the IC never goes nonphysical
        const real_t temp_safe = std::max(temp, 0.2 * temp_inf);

        // CPG isentropic relations used only to generate the spatial field
        const real_t den  = rho_inf * std::pow(temp_safe / temp_inf, 1.0 / gm1);
        const real_t pres = pres_inf * std::pow(temp_safe / temp_inf, gamma / gm1);

        const real_t rhoe = pres / gm1;
        const real_t rhoE = rhoe + 0.5 * den * vel2;

        y(0) = den;
        y(1) = den * velX;
        y(2) = den * velY;
        y(3) = rhoE;
    };
}

// Registration helper that automatically registers these functions
struct RegisterLTEVortex
{
    RegisterLTEVortex()
    {
        // Register initial condition.
        ConditionFactory::Instance().RegisterInitialCondition5("LTEVortexIC", LTEVortexIC);
    }
};
// Global static instance to ensure registration happens at startup.
static RegisterLTEVortex regLTEVortex;

}