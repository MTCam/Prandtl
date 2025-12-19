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
    mu = gasModel->viscosity(S);
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

    real_t vx = S.velocity_x();
    real_t vy = S.velocity_y();
    real_t vz = S.velocity_z();

    gasModel->grad_temperature(3, S, grad_rho, grad_p, grad_t);

    div = dudx + dvdy + dwdz;

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
    flux(4, 2) = vx * flux(1, 2) + vy * flux(2, 2) + vz * flux(3, 2) + kappa * cv_dTdz; 
}

void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, const Vector &dqdy, DenseMatrix &flux) const
{
    PointStateView S{state.GetData(), stateLayout.get()};
    real_t kappa = gasModel->thermal_conductivity(S);
    mu = gasModel->viscosity(S);
    real_t mu_bulk_loc = gasModel->bulk_viscosity(S);    


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
    real_t vx = S.velocity_x();
    real_t vy = S.velocity_y();

    gasModel->grad_temperature(2, S, grad_rho, grad_p, grad_t);
    div = dudx + dvdy;

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk_loc * div);
    flux(2, 0) = mu * (dudy + dvdx);
    flux(3, 0) = vx * flux(1, 0) + vy * flux(2, 0) + kappa * grad_t[0];

    flux(1, 1) = mu * (dvdx + dudy);
    flux(2, 1) = mu * (2.0 * dvdy - mu_bulk_loc * div);
    flux(3, 1) = vx * flux(1, 1) + vy * flux(2, 1) + kappa * grad_t[1];
}

void NavierStokesFlux::ComputeViscousFlux(const Vector &state, const Vector &dqdx, DenseMatrix &flux) const
{
    PointStateView S{state.GetData(), stateLayout.get()};
    real_t kappa = gasModel->thermal_conductivity(S);
    mu = gasModel->viscosity(S);
    real_t mu_bulk_loc = gasModel->bulk_viscosity(S);    

    const real_t &drdx = dqdx(0);
    const real_t &dudx = dqdx(1);
    const real_t &dpdx = dqdx(2);

    const real_t grad_rho[1] = {drdx};
    const real_t grad_p[1] = {dpdx};
    real_t grad_t[1] = {0.0};
    real_t vx = S.velocity_x();
    gasModel->grad_temperature(1, S, grad_rho, grad_p, grad_t);
    div = dudx;

    flux(1, 0) = mu * (2.0 * dudx - mu_bulk_loc * div);
    flux(2, 0) = vx * flux(1, 0) + kappa * grad_t[0];
}

real_t NavierStokesFlux::ComputeInviscidFluxDotN(const Vector &x, const Vector &nor, FaceElementTransformations &Tr, Vector &fluxN) const
{
    return ComputeFluxDotN(x, nor, Tr, fluxN);
}


}
