# Prandtl Developer Notes  
*(Draft — Phase 1: State Layout & Accessor Work)*

## 1. Purpose and Scope

This document provides a developer-oriented overview of Prandtl’s core data structures, state layout, and MFEM integration. It serves as a practical reference during Phase 1 of the modernization plan (state accessor work and device-readiness).  
:contentReference[oaicite:0]{index=0}

### Goals of This Document
- Define the core data types used in the solver (mesh, FE spaces, solution state).  
- Describe the solution state layout (conserved variables, block structure).  
- Establish invariants and requirements for the new `StateView` accessor.  
- Link these concepts to the call sequence and MFEM abstractions.  
- Provide a stable base for Phase 2 work (GasModel/EOS, PA/GPU paths).

### What This Document Is Not
- Not a full tutorial for MFEM.  
- Not a numerics text — only the connections to implementation.  
- Not a frozen specification; it is updated as part of modernization work.

---

## 2. Core Data Types and State Layout

This section captures the minimal but authoritative details required to refactor Prandtl’s state access into a structured, GPU-safe accessor (`StateView`).  

### 2.1 Key Simulation Data Types

| Concept | Type | Location in Code | Description |
|--------|------|------------------|-------------|
| Parallel mesh | `mfem::ParMesh` | `Simulation.cpp` | Distributed mesh used throughout DGSEM operators. |
| DG finite element collection | `mfem::DG_FECollection` | `Simulation.cpp` | FE basis definition for both scalar and system components. |
| Scalar FE space | `mfem::ParFiniteElementSpace` | `fes` | DG space for scalar fields (e.g. density). |
| System FE space | `mfem::ParFiniteElementSpace` | `vfes` | DG space with `vdim = num_equations`, storing conserved state. |
| Solution state | `mfem::ParGridFunction` | `sol` | Global vector containing all conserved DOFs. |
| DG operator | `Prandtl::DGSEMOperator` | `DGSEMOperator.*` | MFEM `TimeDependentOperator` implementing RHS evaluation. |
| DG nonlinear form | `Prandtl::DGSEMNonlinearForm` | `DGSEMNonlinearForm.*` | Spatial discretization (volume and face integrators). |

Examples of their instantiation appear throughout `Simulation.cpp`.  
:contentReference[oaicite:1]{index=1}

---

### 2.2 The Conserved Solution State (`sol`)

Prandtl stores the solution in a single MFEM `ParGridFunction` called `sol`.  
This object encapsulates a global vector of length:

```cpp
num_docs_system = vfes->GetVSize()
```


where:

- `vfes` is the system FE space (`vdim = num_equations`)
- `num_dofs_scalar = fes->GetNDofs()`
- `num_dofs_system = num_equations * num_dofs_scalar`

Internally, the data can be accessed via:

```cpp
real_t *sol_state = sol->GetData();
```

The layout is equation-blocked:

```cpp
real_t rho  = sol_state[0*num_dofs_scalar + i];
real_t rhoU = sol_state[1*num_dofs_scalar + i];
real_t rhoV = sol_state[2*num_dofs_scalar + i];
real_t E    = sol_state[3*num_dofs_scalar + i];
```

#### Conserved State Block Ordering
1. Density  
2. Momentum x  
3. Momentum y (if `dim > 1`)  
4. Momentum z (if `dim > 2`)  
5. Total energy  

Future extensions add additional blocks:
- Passive scalars  
- Species  
- LTE internal variables  

The accessor must not hard-code the number of blocks.

---

### 2.3 Required Invariants for State Layout

1. **Equation-blocked storage**  
   `U = [block0 | block1 | block2 | …]`, each of size `num_dofs_scalar`.

2. **All components share the same FE space**  
   DOFs per component are identical.

3. **Component ordering is fixed and globally relied upon**  
   (DG volume, face terms, BCs, fluxes, post-processing).

4. **GPU-friendly layout**  
   Access must resolve to simple integer arithmetic.

5. **Device and host access must use identical indexing rules**  
   Enforced through `StateView`.

---

### 2.4 Design Goals for the `StateView` / State Accessor

The new accessor will replace all raw indexing of the solution vector.

#### Functional Requirements
- Named accessors:
  - `rho(i)`
  - `mom_x(i)`, `mom_y(i)`, `mom_z(i)`
  - `energy(i)`
- Access by:
  - DOF index (`i`)
  - Block index and offset (`block, i`)
  - Full block spans

#### Structural Requirements
- Header-only and inlinable.  
- No virtual dispatch.  
- Accepts:
  - Underlying pointer (`real_t*`)  
  - `num_dofs_scalar`  
  - Dimensions (`dim`)  
  - Number of equations  
- Must not include EOS or primitive conversions.

#### Non-Goals
- Not responsible for derived variables.  
- Not responsible for GasModel/EOS.  
- Not responsible for memory ownership.

---

### 2.5 Planned Extensions (Future-Proofing)

Future tracks (LTE, NLTE, mixture transport) will add more components.  
Therefore the accessor must:

- Compute block offsets from metadata.  
- Support arbitrary additional variables without rewrite.  
- Centralize ordering and naming in one place.

---
## 3. Simulation Workflow and Call Sequence

This section summarizes how a Prandtl simulation is constructed and advanced in time, and how the main MFEM and Prandtl objects interact during this process.

### 3.1 Big-Picture Simulation Workflow

At a high level, a Prandtl run follows this sequence:

1. **Read runtime configuration**
   - Parse input JSON.
   - Construct `PhysicsConstants` (e.g. `gamma`, `Pr`, `R_gas`, viscosity parameters).

2. **Build mesh and parallel mesh**
   - Load a serial `mfem::Mesh`.
   - Construct a parallel `mfem::ParMesh` and apply optional parallel refinements (`par_ref_levels`).

3. **Create finite element collections and spaces**
   - Build a DG finite element collection (`DG_FECollection`).
   - Create scalar and system parallel FE spaces:
     - Scalar space `fes` for scalar fields.
     - System space `vfes` for the conserved state (with `vdim = num_equations`).

4. **Initialize the solution state**
   - Allocate the solution `ParGridFunction` `sol` on `vfes`.
   - Either:
     - Load from checkpoint, or
     - Project an initial condition coefficient onto `sol`.

5. **Construct fluxes and DG operator**
   - Create the physical flux object (e.g. `NavierStokesFlux`).
   - Wrap it in a numerical flux (e.g. `ChandrashekarFlux`).
   - Build the `DGSEMOperator`, which owns the DGSEM nonlinear form and related data.

6. **Configure boundary conditions and indicators**
   - Add appropriate boundary face integrators to the operator (symmetry, walls, inflow/outflow, etc.).
   - Initialize limiters, indicators, and any FV blending coefficients.

7. **Set up time integration**
   - Create an MFEM ODE solver (e.g. `ForwardEulerSolver`, RK variants).
   - Initialize the ODE solver with `DGSEMOperator` as the `TimeDependentOperator`.

8. **Time-stepping loop**
   - Optionally compute a CFL-based time step.
   - Call `ode_solver->Step(*sol, t, dt)` to advance the state.
   - Update diagnostics, compute derived quantities, and write visualization output.

Subsequent subsections detail the RHS call chain, where and how the conserved state is accessed, and how these pieces connect to MFEM’s abstractions.

### 3.2 RHS Call Chain  
*(ODE Solver → DGSEMOperator → DGSEMNonlinearForm)*

This subsection describes how a time step evaluates the spatial discretization and produces the time derivative `dU/dt`. It connects the MFEM ODE solvers with Prandtl’s DGSEM implementation.

#### 3.2.1 MFEM ODE Solver Entry Point

All MFEM explicit ODE solvers call the `Mult` method of the registered `TimeDependentOperator`.  
For example, the forward Euler solver performs:

```cpp
f->SetTime(t);
f->Mult(x, dxdt);
x.Add(dt, dxdt);
```
Here:
- f is the DGSEMOperator  
- x is the current solution vector (sol)  
- dxdt becomes the right-hand-side vector (dU/dt)  

The ODE solver does not inspect the PDE; it simply forwards the call to the operator.

#### 3.2.2 DGSEMOperator::Mult

DGSEMOperator::Mult(u, dudt) is the top-level PDE evaluation routine.  
It performs the following steps (conceptually):

1. Select the active state  
   - In axisymmetric mode, convert the weighted state back into U.  
   - Otherwise, Ustate = u.  
   This is the first point in the call chain where the conserved state layout matters.

2. Optional subcell FV blending  
   - Compute or update FV blending coefficients if blending is enabled.

3. Optional parabolic and viscous preparation  
   - Compute entropy-related vectors.  
   - Compute primitive gradients using lifting operators.  
   - Prepare gradient information for viscous flux evaluation.

4. Delegate spatial discretization  
   - Call the DGSEMNonlinearForm to evaluate volume, face, and boundary contributions.  
   - Convective-only cases call a simple Mult(Ustate, dudt).  
   - Parabolic cases call the overload that also receives gradient data.

5. Final accumulation  
   - At return, dudt contains the full spatial operator result.

DGSEMOperator is also the place where scattered raw indexing into the conserved state currently appears (density, momentum components, and total energy). All of these will be redirected through StateView during Phase 1.

#### 3.2.3 DGSEMNonlinearForm::Mult

DGSEMNonlinearForm handles the detailed DGSEM spatial discretization.  
Mult(U, dudt) performs:

- Volume contributions on each element  
- Interior face fluxes between local elements  
- Neighbor face contributions using MFEM's face neighbor exchange  
- Boundary face integrators for user-specified BCs  
- Optional viscous or parabolic flux terms  

The method:
- Gathers DOFs corresponding to each element or face  
- Pulls neighbor DOFs when required  
- Applies numerical fluxes and integrators  
- Accumulates each contribution into the global dudt vector  

DGSEMNonlinearForm is the main consumer of the conserved state and therefore one of the primary targets for StateView integration.

### 3.3 Volume Contributions (Element Interior Terms)

The first major component of DGSEMNonlinearForm::Mult is the evaluation of 
element-interior DGSEM volume terms. These terms assemble the divergence of 
the inviscid (and optionally viscous) fluxes within each element.

Conceptually, the volume loop performs the following steps for every element:

1. Gather the element's DOFs from the global solution vector U.  
2. Evaluate the solution at quadrature points.  
3. Compute fluxes at each quadrature point using the chosen numerical flux 
   (e.g., Navier–Stokes flux for Euler/CNS).  
4. Apply derivative and interpolation operators associated with the DGSEM basis.  
5. Assemble the contribution to dudt for the current element.

Typical code paths follow this pattern:

```
- el_u = gather element DOFs  
- compute fluxes F(U) at quadrature points  
- apply DGSEM differentiation matrices  
- accumulate local dudt contribution  
```

The key observation for modernization work is:

- All accesses to element DOFs currently use raw indexing based on the global 
  conserved state layout.
- These accesses will be replaced by the StateView API.
- The DGSEM volume loop is a key GPU hot spot; eliminating indirect indexing 
  and virtual calls here is crucial for device execution.

In subsequent subsections, interior face terms and boundary face terms are described,
completing the RHS spatial contributions.

### 3.4 Interior Face Terms (Shared Interfaces Between Elements)

The second major component of DGSEMNonlinearForm::Mult is the evaluation of
interior face contributions. These enforce conservation and stability across
element interfaces by applying a numerical flux.

Interior face processing conceptually follows this sequence:

1. Identify all interior faces of the mesh.  
   - In parallel runs, MFEM splits these into local faces and shared (MPI) faces.

2. For each interior face:
   - Fetch the DOFs of the "left" element.
   - Fetch the DOFs of the "right" element (neighbor element).  
     In parallel: pull data from face neighbor buffers.

3. Transform DOFs to the face reference frame using MFEM's face transformations.

4. Evaluate the numerical flux at quadrature points using states from both sides.

5. Assemble flux contributions into the dudt vector for the left element.

The pattern can be summarized as:

```cpp
for each interior face:
    U_left  = gather DOFs for element on side 1
    U_right = gather DOFs for neighbor element
    compute numerical flux = F_hat(U_left, U_right)
    accumulate contribution into dudt for element 1
```

Key observations for modernization:

- Interior face terms consume a substantial fraction of runtime and are highly
  sensitive to memory access patterns.
- MFEM's face neighbor exchange populates U_right for shared faces; these accesses
  must be consistent with the StateView abstraction.
- Numerical flux evaluation calls into Navier–Stokes flux objects, which currently
  use raw indexing of density, momentum components, and energy.
- GPU readiness requires eliminating STL, dynamic allocation, and virtual dispatch
  from this loop.

The next subsection describes boundary face terms, which follow a similar pattern
but use specialized integrators for boundary conditions.

