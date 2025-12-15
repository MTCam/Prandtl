#pragma once

#include <vector>
#include <GasState.hpp>

// Adapter that uses the new StateLayout + FieldStateView as a "state-like"
// object compatible with run_basic_mass_momentum_energy_test.
class GasStateSemanticsAdapter
{
public:
    using real_t = Prandtl::real_t;

    GasStateSemanticsAdapter(int dim, int num_dofs_scalar)
        : layout_(dim, num_dofs_scalar),
          data_((dim + 2 + layout_.num_scalars) * num_dofs_scalar),
          view_(data_.data(), &layout_)
    { }

    // --- StateSemantics "contract" API ---

    int dim() const      { return layout_.dim; }
    int num_dofs() const { return layout_.num_dofs_scalar; }

    // Getters
    real_t mass(int i) const
    {
        return view_.mass(i);
    }

    real_t momentum(int d, int i) const
    {
        return view_.momentum(d, i);
    }

    real_t energy(int i) const
    {
        return view_.energy(i);
    }

    // Setters
    void set_mass(int i, real_t rho)
    {
        view_.mass(i) = rho;
    }

    void set_momentum(int d, int i, real_t rho_u_d)
    {
        view_.momentum(d, i) = rho_u_d;
    }

    void set_energy(int i, real_t rhoE)
    {
        view_.energy(i) = rhoE;
    }

private:
    Prandtl::StateLayout layout_;
    std::vector<real_t> data_;
    Prandtl::FieldStateView view_;
};
