#pragma once
// StateLayout and State Views for Gas Simulations
//
// Goal:
//   - Centralize the mapping from (equation, dof) -> flat index
//   - Provide simple per-DOF view for immediate refactoring
//   - Provide general field-level view for future use
//
// Layout assumption (canonical, matches current code):
// Note that potentially we change to rho*Y_alpha.
//   U(eq, dof) is stored as equation-blocked:
//       U = [ rho       (block 0)
//           | rho*u_x   (block 1)
//           | rho*u_y   (block 2, if dim > 1)
//           | rho*u_z   (block 3, if dim > 2)
//           | rho*E     (block dim+1)
//           | scalars... ]
//
// Each block has length = num_dofs_scalar.
//
// Notes:
//   - DofStateView<T>        : per-DOF view, simple and minimal, for refactors.
//   - FieldStateView<T>      : field-level view (full vector), forward-looking.
//
// Refactoring plan:
//
//   1) Replace raw indexing with DofStateView first.
//   2) Introduce FieldStateView in loops or where convenient/needed.
//   3) Extend StateLayout with scalars/species, without touching callers.
//

#ifndef MFEM_HOST_DEVICE
#define MFEM_HOST_DEVICE
#endif

namespace Prandtl
{

using real_t = double;

// -----------------------------------------------------------------------------
// StateLayout: single point to define how state storage is arranged
// -----------------------------------------------------------------------------
struct StateLayout
{
    int dim;              // spatial dimension (1,2,3)
    int num_dofs_scalar;  // DOFs per scalar field (block length)

    // Equation indices (0-based)
    int eq_mass;          // mass density
    int eq_mom[3];        // momentum density components: x,y,z
    int eq_energy;        // total energy density

    // Optional scalar support (forward-looking)
    int eq_scalar0;       // index of first scalar component (or -1 if none)
    int num_scalars;      // number of scalar components

    MFEM_HOST_DEVICE
    StateLayout()
        : dim(0),
          num_dofs_scalar(0),
          eq_mass(0),
          eq_energy(0),
          eq_scalar0(-1),
          num_scalars(0)
    {
        eq_mom[0] = eq_mom[1] = eq_mom[2] = -1;
    }

    // Canonical ordering:
    //   [rho, rho*u_0, ..., rho*u_(dim-1), rho*E, scalars...]
    MFEM_HOST_DEVICE
    StateLayout(int dim_,
                int num_dofs_scalar_,
                int num_scalars_ = 0)
        : dim(dim_),
          num_dofs_scalar(num_dofs_scalar_),
          eq_mass(0),
          eq_energy(dim_ + 1),
          eq_scalar0(num_scalars_ > 0 ? (dim_ + 2) : -1),
          num_scalars(num_scalars_)
    {
        // Momentum components follow mass
        for (int d = 0; d < 3; ++d)
        {
            eq_mom[d] = (d < dim_) ? (1 + d) : -1;
        }
    }

    // Flat index into the equation-blocked vector
    MFEM_HOST_DEVICE inline int index(int equation, int dof) const
    {
        return equation * num_dofs_scalar + dof;
    }
};


// -----------------------------------------------------------------------------
// DofStateView<T>: per-DOF view (immediate refactor tool).
//
// Usage pattern:
//
//   const double* U = sol->Read();
//   StateLayout layout(dim, num_dofs_scalar);
//   for (int i = 0; i < num_dofs_scalar; ++i) {
//       DofStateView<const double> S{U, &layout, i};
//       double rho  = S.mass();
//       double rhoU = S.momentum(0);
//       double E    = S.energy();
//       ...
//   }
//
// This is a small, value-type-like view with correct indexing.
// -----------------------------------------------------------------------------
template <typename T>
struct DofStateView
{
    const T* U;                // pointer into equation-blocked storage
    const StateLayout* L;      // layout metadata
    int dof;                   // which DOF (0 .. num_dofs_scalar-1)

    MFEM_HOST_DEVICE
    DofStateView()
        : U(nullptr), L(nullptr), dof(0)
    { }

    MFEM_HOST_DEVICE
    DofStateView(const T* U_,
                 const StateLayout* L_,
                 int dof_)
        : U(U_), L(L_), dof(dof_)
    { }

    MFEM_HOST_DEVICE inline bool is_valid() const
    {
        return (U != nullptr) && (L != nullptr);
    }

    // Mass / density
    MFEM_HOST_DEVICE inline T mass() const
    {
        return U[ L->index(L->eq_mass, dof) ];
    }

    // Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline T momentum(int d) const
    {
        return U[ L->index(L->eq_mom[d], dof) ];
    }

    MFEM_HOST_DEVICE inline T momentum_x() const
    {
        return momentum(0);
    }

    MFEM_HOST_DEVICE inline T momentum_y() const
    {
        return (L->dim > 1) ? momentum(1) : T(0);
    }

    MFEM_HOST_DEVICE inline T momentum_z() const
    {
        return (L->dim > 2) ? momentum(2) : T(0);
    }

    // Total energy
    MFEM_HOST_DEVICE inline T energy() const
    {
        return U[ L->index(L->eq_energy, dof) ];
    }

    // Scalar components, if present
    MFEM_HOST_DEVICE inline T scalar(int k) const
    {
        return U[ L->index(L->eq_scalar0 + k, dof) ];
    }
};


// -----------------------------------------------------------------------------
// FieldStateView<T>: Field-level view.
//
// Wraps a T* + StateLayout and provides equation-blocked access over all DOFs.
// Use in loops that already use (eq, i) patterns, or when needing more general
// mechanism than DofStateView.
// -----------------------------------------------------------------------------
template <typename T>
struct FieldStateView
{
    T* data;                     // raw pointer to equation-blocked storage
    const StateLayout* layout;   // layout metadata

    MFEM_HOST_DEVICE
    FieldStateView()
        : data(nullptr), layout(nullptr)
    { }

    MFEM_HOST_DEVICE
    FieldStateView(T* data_,
                   const StateLayout* layout_)
        : data(data_), layout(layout_)
    { }

    MFEM_HOST_DEVICE inline bool is_valid() const
    {
        return (data != nullptr) && (layout != nullptr);
    }

    // Generic access by (equation, dof)
    MFEM_HOST_DEVICE inline T& u(int equation, int dof) const
    {
        return data[ layout->index(equation, dof) ];
    }

    // Named accessors for convenience
    MFEM_HOST_DEVICE inline T& mass(int dof) const
    {
        return u(layout->eq_mass, dof);
    }

    MFEM_HOST_DEVICE inline T& momentum(int component, int dof) const
    {
        return u(layout->eq_mom[component], dof);
    }

    MFEM_HOST_DEVICE inline T& momentum_x(int dof) const
    {
        return u(layout->eq_mom[0], dof);
    }

    MFEM_HOST_DEVICE inline T& momentum_y(int dof) const
    {
        return u(layout->eq_mom[1], dof);
    }

    MFEM_HOST_DEVICE inline T& momentum_z(int dof) const
    {
        return u(layout->eq_mom[2], dof);
    }

    MFEM_HOST_DEVICE inline T& energy(int dof) const
    {
        return u(layout->eq_energy, dof);
    }

    MFEM_HOST_DEVICE inline T& scalar(int k, int dof) const
    {
        return u(layout->eq_scalar0 + k, dof);
    }
};
} // namespace Prandtl
