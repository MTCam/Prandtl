#pragma once

#include "prandtl_kernels.hpp"

namespace Prandtl
{

  struct PhysicsConstants
  {
    const real_t gamma;
    const real_t gammaInverse;
    const real_t gammaP1;
    const real_t gammaM1;
    const real_t gammaP1Inverse;
    const real_t gammaM1Inverse;
    const real_t gamma_gammaM1Inverse; // gamma * gammaM1Inverse;
    const real_t gammaM1_gammaInverse; // gammaM1 * gammaInverse;

    const real_t Pr;
    const real_t PrInverse;
    const real_t R_gas;
    const real_t cp;
    const real_t mu;

    PhysicsConstants(real_t gamma, real_t Pr, real_t R_gas, real_t mu)
      : gamma(gamma), Pr(Pr), R_gas(R_gas), mu(mu),
        gammaInverse(1.0 / gamma), gammaM1(gamma - 1.0), gammaP1(gamma + 1.0),
        gammaM1Inverse(1.0 / gammaM1), gammaP1Inverse(1.0 / gammaP1),
        gammaM1_gammaInverse(gammaM1 * gammaInverse), gamma_gammaM1Inverse(gamma * gammaM1Inverse),
        PrInverse(1.0 / Pr), cp(gamma_gammaM1Inverse * R_gas) {}

    const real_t mu_bulk = 2.0 / 3.0;
    const real_t mu0 = 1.716e-5;
    const real_t T0 = 273.15;
    const real_t Ts = 110.4;
};

// extern const real_t gamma;
// extern const real_t gammaInverse;
// extern const real_t gammaP1;
// extern const real_t gammaM1;
// extern const real_t gammaP1Inverse;
// extern const real_t gammaM1Inverse;
// extern const real_t gamma_gammaM1Inverse; // gamma * gammaM1Inverse;
// extern const real_t gammaM1_gammaInverse; // gammaM1 * gammaInverse; 

// extern const real_t Pr;
// extern const real_t PrInverse;

// extern const real_t R_gas;
// extern const real_t cp;

// extern const real_t mu;

// const real_t mu_bulk = 2.0 / 3.0;
// const real_t mu0 = 1.716e-5;
// const real_t T0 = 273.15;
// const real_t Ts = 110.4;
// const real_t T0pTs = T0 + Ts;


}
