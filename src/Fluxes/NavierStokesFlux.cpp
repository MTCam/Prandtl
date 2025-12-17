#include "NavierStokesFlux.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{

real_t NavierStokesFlux::ComputeInviscidFlux(const Vector &state, ElementTransformation &Tr, DenseMatrix &flux) const
{
    return ComputeFlux(state, Tr, flux);
}

void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, DenseMatrix &flux) const
{
    PointStateView S{state.GetData(), stateLayout.get()};
    Conserv2Prim(state, prim, gammaM1);

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

#ifdef SUTHERLAND
    temperature = gasModel->eos.temperature()
    mu = gasModel->transport.viscosity(temperature)
#endif
    real_t Cp = gasModel->eos.cp(S);
    lambda = gasModel->transport.thermal_conductivity(mu, Cp); //  * gamma * PrInverse;
    lambda = mu * gamma * PrInverse;
    std::cout << "1: mu = " << mu << ",Cp = " << Cp << ",lambda = " << lambda << std::endl;

    div = dudx + dvdy + dwdz;
    cv_dTdx = gammaM1Inverse / prim(0) * (dpdx - prim(num_equations - 1) / prim(0) * drdx);
    cv_dTdy = gammaM1Inverse / prim(0) * (dpdy - prim(num_equations - 1) / prim(0) * drdy);
    cv_dTdz = gammaM1Inverse / prim(0) * (dpdz - prim(num_equations - 1) / prim(0) * drdz);

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk * div);
    flux(2, 0) = mu * (dudy + dvdx);
    flux(3, 0) = mu * (dudz + dwdx);
    flux(4, 0) = prim(1) * flux(1, 0) + prim(2) * flux(2, 0) + prim(3) * flux(3, 0) + lambda * cv_dTdx;

    flux(1, 1) = mu * (dvdx + dudy);
    flux(2, 1) = mu * (2.0 * dvdy - mu_bulk * div);
    flux(3, 1) = mu * (dvdz + dwdy);
    flux(4, 1) = prim(1) * flux(1, 1) + prim(2) * flux(2, 1) + prim(3) * flux(3, 1) + lambda * cv_dTdy;

    flux(1, 2) = mu * (dwdx + dudz);
    flux(2, 2) = mu * (dwdy + dvdz);
    flux(3, 2) = mu * (2.0 * dwdz - mu_bulk * div);
    flux(4, 2) = prim(1) * flux(1, 2) + prim(2) * flux(2, 2) + prim(3) * flux(3, 2) + lambda * cv_dTdz; 
}

void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, DenseMatrix &flux) const
{
    Conserv2Prim(state, prim, gammaM1);
    PointStateView S{state.GetData(), stateLayout.get()};

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

#ifdef SUTHERLAND
    temperature = gasModel->eos.temperature()
    mu = gasModel->transport.viscosity(temperature)
#endif

    real_t Cp = gasModel->eos.cp(S);
    real_t kappa = gasModel->transport.thermal_conductivity(mu, Cp);
    gasModel->eos.grad_temperature(2, S, grad_rho, grad_p, grad_t);

    div = dudx + dvdy;

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk * div);
    flux(2, 0) = mu * (dudy + dvdx);
    flux(3, 0) = prim(1) * flux(1, 0) + prim(2) * flux(2, 0) + kappa * grad_t[0];

    flux(1, 1) = mu * (dvdx + dudy);
    flux(2, 1) = mu * (2.0 * dvdy - mu_bulk * div);
    flux(3, 1) = prim(1) * flux(1, 1) + prim(2) * flux(2, 1) + kappa * grad_t[1];
}

void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, DenseMatrix &flux) const
{
    Conserv2Prim(state, prim, gammaM1);
    PointStateView S{state.GetData(), stateLayout.get()};

    const real_t &drdx = dqdx(0);
    const real_t &dudx = dqdx(1);
    const real_t &dpdx = dqdx(2);

#ifdef SUTHERLAND
    temperature = gasModel->eos.temperature()
    mu = gasModel->transport.viscosity(temperature)
#endif

    real_t Cp = gasModel->eos.cp(S);
    lambda = gasModel->transport.thermal_conductivity(mu, Cp); //  * gamma * PrInverse;
    lambda = mu * gamma * PrInverse;
    std::cout << "3: mu = " << mu << ",Cp = " << Cp << ",lambda = " << lambda << std::endl;
    div = dudx;
    cv_dTdx = gammaM1Inverse / prim(0) * (dpdx - prim(num_equations - 1) / prim(0) * drdx);

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk * div);
    flux(2, 0) = prim(1) * flux(1, 0) + lambda * cv_dTdx;
}

real_t NavierStokesFlux::ComputeInviscidFluxDotN(const Vector &x, const Vector &nor, FaceElementTransformations &Tr, Vector &fluxN) const
{
    return ComputeFluxDotN(x, nor, Tr, fluxN);
}


}
