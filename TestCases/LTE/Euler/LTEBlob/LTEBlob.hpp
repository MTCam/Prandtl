#pragma once

#include "ConditionFactory.hpp"

namespace Prandtl
{

// LTE Blob initial condition
std::function<void(const Vector&, Vector&)> LTEBlobIC(real_t radius,
                                                        real_t T_inf,
                                                        real_t T_blob,
                                                        real_t P_inf)
{
    return [=](const Vector &x, Vector &y)
    {
        MFEM_ASSERT(x.Size() == 2, "");

        const real_t xc = 0.0;
        const real_t yc = 0.0;

        real_t dx = x(0) - xc;
        real_t dy = x(1) - yc;

        real_t r2rad = (dx*dx + dy*dy) / (radius * radius);

        const real_t exp_full = std::exp(-r2rad);

        // Quiescent velocity field
        const real_t velX = 0.0;
        const real_t velY = 0.0;
        const real_t vel2 = velX * velX + velY * velY;

        // Gaussian temperature perturbation
        const real_t T = T_inf + (T_blob - T_inf) * exp_full;

        real_t R_gas = 287.05;
        const real_t gamma = 1.4;

        const real_t den   = P_inf / (R_gas * T);
        const real_t rhoe  = P_inf / (gamma-1.0);
        const real_t rhoE  = rhoe + 0.5 * den * vel2;

        y(0) = den;
        y(1) = den * velX;
        y(2) = den * velY;
        y(3) = rhoE;
    };
}

// Registration helper that automatically registers these functions
struct RegisterLTEBlob
{
    RegisterLTEBlob()
    {
        // Register initial condition.
        ConditionFactory::Instance().RegisterInitialCondition4("LTEBlobIC", LTEBlobIC);
    }
};
// Global static instance to ensure registration happens at startup.
static RegisterLTEBlob regLTEBlob;

}