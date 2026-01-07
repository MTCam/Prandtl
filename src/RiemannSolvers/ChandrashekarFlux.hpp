#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"

namespace Prandtl
{

class ChandrashekarFlux : public NumericalFlux
{
private:
    mutable Vector metric;
  std::shared_ptr<const IdealGasModel> gasModel;
  std::shared_ptr<const StateLayout> stateLayout;
public:
  ChandrashekarFlux(const NavierStokesFlux &fluxFunction);
  real_t ComputeFaceFlux(const Vector &state1, const Vector &state2, const Vector &nor, Vector &flux) const override;
  real_t ComputeVolumeFlux(const Vector &state1, const Vector &state2, const Vector &metric1, const Vector &metric2, Vector &F_tilde) override;

};

}
