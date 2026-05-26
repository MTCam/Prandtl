#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"

namespace Prandtl
{

using namespace mfem;

class Indicator
{
protected:
    std::shared_ptr<ParFiniteElementSpace> vfes;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::shared_ptr<ParGridFunction> eta;

    Array<int> vdof_indices, ind_indx;
    Vector el_vdofs, ind_dof;
    int num_equations, ndofs, order, dim;
    Vector state;
public:
    Indicator(std::shared_ptr<ParFiniteElementSpace> vfes, std::shared_ptr<ParFiniteElementSpace> fes0, std::shared_ptr<ParGridFunction> eta);
    virtual void CheckSmoothness(const Vector &x, const ThermoTablesView &thermoTables) = 0;
    virtual void CheckIndicatorSmoothness(const Vector &indicator) = 0;
    virtual ~Indicator() = default;
};
}
