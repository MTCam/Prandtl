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

This section summarizes how a Prandtl simulation is constructed and advanced in time, and how the main MFEM and Prandtl objects interact during this process. This section also summarizes how the conserved state is created, passed through the
operator hierarchy, and consumed by the DGSEM kernels.

At a high level:

1. The solution state lives in the ParGridFunction `sol`, defined on the
   system finite element space `vfes`. It stores all conserved variables
   using the equation-blocked layout described in Section 2.

2. During each time step, the MFEM ODE solver calls  
   DGSEMOperator::Mult(sol, dudt).  
   This is the top-level RHS routine and the first consumer of the conserved
   state.

3. DGSEMOperator delegates the spatial discretization to  
   DGSEMNonlinearForm::Mult(...).  
   This routine evaluates all DG volume, interior face, boundary face, and MPI
   shared-face contributions, where the conserved state is accessed heavily.

The subsections that follow provide detail on each of these stages and how they
relate to Prandtl’s data structures and MFEM’s abstractions.

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


### 3.5 Boundary Face Terms and Boundary Integrators

Boundary faces are handled similarly to interior faces, but instead of using
a left–right pair of element states, the DGSEMNonlinearForm applies a boundary
condition integrator. Each integrator encodes how the "exterior" state should
be constructed (e.g., reflective, inflow, no-slip, adiabatic).

Conceptually, boundary face processing follows this pattern:

1. Loop over all boundary faces in the mesh.  
2. For each face, identify the boundary condition type from the input file.  
3. Gather DOFs for the adjacent element (the interior side).  
4. Create or infer the "ghost" exterior state based on the boundary condition.  
5. Evaluate the numerical flux between interior and ghost states.  
6. Accumulate the contribution into the dudt vector for the adjacent element.

This can be summarized as:

```cpp
for each boundary face:
    U_int = gather DOFs for the element
    U_ext = apply boundary condition rule to construct ghost state
    compute numerical flux = F_hat(U_int, U_ext)
    accumulate contribution into dudt
```

Examples of boundary integrators wired into the operator include:

- Symmetry boundary (reflective or axisymmetric)
- No-slip adiabatic wall
- Generic inflow and outflow conditions
- Axisymmetric boundary conditions (enabled by compile-time flag AXISYMMETRIC)

The DGSEMOperator registers these integrators using MFEM mechanisms such as 
AddBdrFaceIntegrator, which adds the boundary integrator along with a marker 
identifying which boundary faces it applies to.

Modernization considerations:

- Each boundary integrator contains raw accesses into the conserved state 
  to compute velocity components, pressure, or temperature.
- These access patterns must be rewritten to use the StateView API.
- Many BC implementations also access physics constants directly; these 
  will eventually route through the GasModel abstraction.
- GPU execution requires that boundary face integrators avoid dynamic 
  allocation and virtual dispatch inside hot loops.

The next subsection completes the RHS assembly by describing the MPI 
face-neighbor data exchange required for parallel runs.

### 3.6 Parallel Execution: Face Neighbor Exchange (MPI)

In parallel runs, DGSEMNonlinearForm must evaluate numerical fluxes on faces 
shared between elements that reside on different MPI ranks. MFEM provides a 
built-in mechanism for exchanging the necessary degrees of freedom through 
its face-neighbor data structures.

The sequence for parallel face exchange proceeds as follows:

1. Identify all shared faces  
   - MFEM partitions the mesh so each rank knows which faces touch off-rank 
     elements.

2. Pack local element DOFs into a contiguous send buffer  
   - Each field involved in exchange (e.g., Ustate and gradients when viscous 
     terms are enabled) is packed into MFEM’s "face neighbor" buffers.

3. Exchange neighbor data using nonblocking MPI communication  
   - MFEM handles the communication through its ParGridFunction and 
     ParFiniteElementSpace infrastructure.
   - After communication completes, the neighbor DOFs appear in 
     FaceNbrData buffers.

4. For each shared face:
   - Retrieve DOFs for the on-rank element.
   - Retrieve DOFs for the off-rank element from the FaceNbrData buffer.
   - Apply the same numerical flux procedure as for interior faces.

5. Accumulate contributions into dudt for the local element.  
   - Only the local element is updated; the neighbor element receives 
     its own contributions on its own rank.

This process can be summarized as:

```cpp
pack local DOFs into face-neighbor send buffer
exchange buffers across MPI ranks
for each shared face:
    U_left  = gather DOFs for the local element
    U_right = gather DOFs from FaceNbrData
    compute numerical flux = F_hat(U_left, U_right)
    update dudt for the local element only
```

Modernization considerations:

- All packing and unpacking of DOFs currently relies on the raw conserved 
  state layout; this must be routed through StateView.
- MPI exchange paths must avoid host-only constructs when GPU execution 
  is enabled; no STL containers or dynamic allocation inside the loops.
- FaceNbrData access on device is possible only if integrators are properly 
  device-enabled and MFEM’s memory mode is set appropriately.
- The accessor must support reading DOFs from either local Ustate or neighbor 
  buffers without branching logic scattered across the code.

With interior faces, boundary faces, and MPI exchanges covered, the next 
subsection completes the spatial operator pipeline by outlining the 
post-assembly update into the global dudt vector.


## 4. Code Navigation: Where to Look in the Source

This section is a practical index into the Prandtl codebase. It highlights
the main locations where the conserved state is accessed, where MFEM
constructs are wired together, and where upcoming refactors (StateView,
GasModel, GPU enablement) will primarily land.

The paths and function names here are meant as stable landmarks, not exact
line numbers.

---

### 4.1 High-Value Locations for StateView Refactor

These are the most important places to inspect and update when replacing
raw indexing with the StateView accessor.

- **Solution state layout and initial debugging**
  - File: `Simulation.cpp`  
  - What to look for:
    - Construction of `vfes`, `fes`, and `sol`.  
    - Definitions of `num_dofs_scalar` and `num_dofs_system`.  
    - Debug blocks that directly call `sol->GetData()` and then compute
      rho, momentum components, and energy using expressions like  
      `sol_state[eq*num_dofs_scalar + i]`. :contentReference[oaicite:0]{index=0}  
  - Why it matters:
    - This code documents the implicit layout assumptions in a concentrated,
      easy-to-read location.

- **Top-level RHS operator: DGSEMOperator**
  - File: `src/Operators/DGSEMOperator.cpp` (and header)  
  - What to look for:
    - The method `DGSEMOperator::Mult(const Vector &u, Vector &dudt)`. :contentReference[oaicite:1]{index=1}  
    - Any use of density, momentum, energy, entropy, or primitive variables
      extracted directly from the solution vector.
    - Axisymmetric-specific code paths (`AXISYMMETRIC` conditional code).
    - Subcell FV blending setup (`SUBCELL_FV_BLENDING`), which may read from
      the conserved state.
  - Why it matters:
    - This is the first consumer of `sol` in the RHS chain and a natural
      place to introduce StateView as the canonical way to interpret the
      solution vector.

- **DGSEM spatial discretization: DGSEMNonlinearForm**
  - File: `src/Operators/NonlinearForm/DGSEMNonlinearForm.cpp`  
  - What to look for:
    - `DGSEMNonlinearForm::Mult(...)` and any helper routines that:
      - Gather element DOFs into local vectors.  
      - Gather face DOFs for interior and shared faces.  
      - Access or interpret conserved variables (rho, rho*u, rho*E, etc.). :contentReference[oaicite:2]{index=2}  
  - Why it matters:
    - This is the primary DGSEM kernel implementation and a major hotspot.
      All DOF access here needs to be compatible with device execution and
      should go through StateView.

- **Boundary integrators**
  - Files: `src/Operators/Boundary/*` (or equivalent boundary integrator
    directory)  
  - What to look for:
    - BC classes registered via `NS->AddBdrFaceIntegrator(...)` in
      `Simulation.cpp` (e.g., symmetry, no-slip adiabatic walls, axisymmetry). :contentReference[oaicite:3]{index=3}  
    - Direct extraction of primitive variables (velocity components, pressure,
      temperature) from the conserved state using raw indexing.
  - Why it matters:
    - Boundary conditions frequently duplicate state access logic; StateView
      should become the single source for that mapping so BCs stay correct
      when state layout evolves.

- **Postprocessing and visualization**
  - File: `Simulation.cpp`  
  - What to look for:
    - Creation of derived fields `rho`, `mom`, `energy`, velocities `u, v, w`,
      and pressure `p` from `sol`.  
    - Loops that compute:  
      - `rho(i)`  
      - `mom(i)` and `mom(i + num_dofs_scalar)`  
      - `energy(i)` and `p(i)` with expressions like  
        `gammaM1 * (energy(i) - 0.5*rho(i)*V_sq)`. :contentReference[oaicite:4]{index=4}  
  - Why it matters:
    - These loops are explicit examples of how the state is interpreted and
      need to remain consistent with whatever StateView defines as the
      canonical ordering and meaning of each block.

### 4.2 MFEM Wiring, Time Integration, and Operator Setup

This subsection points to the places where Prandtl connects its own operators
to MFEM’s abstractions: mesh and FE setup, operator construction, and ODE
solver configuration.

These are key landmarks when you need to understand how everything is glued
together or where to attach new capabilities (e.g., GasModel, PA/GPU paths).

- **Mesh and FE space construction**
  - File: `Simulation.cpp`  
  - What to look for:
    - Creation of `mesh` (serial) and `pmesh` (parallel):  
      construction from input mesh file, followed by `par_ref_levels`
      uniform refinement. :contentReference[oaicite:0]{index=0}  
    - Construction of DG FE collections:
      - `fec`  = DG_FECollection(order, dim, btype)  
      - `fec0` = DG_FECollection(0, dim)
    - Construction of FE spaces:
      - `vfes` (system space, vdim = num_equations)  
      - `fes`  (scalar space)  
      - `dfes` (vector space for gradients, dimension = dim)
  - Why it matters:
    - These constructions determine the size and structure of the solution
      vector and all auxiliary fields (e.g., gradients, indicators).
    - Any future PA/GPU setup that depends on element-wise data will need to
      be consistent with these spaces.

- **DGSEM operator construction**
  - File: `Simulation.cpp`  
  - What to look for:
    - Construction of the physical flux object (`NavierStokesFlux`) using
      `PhysicsConstants` (gamma, Pr, mu, R_gas, etc.). :contentReference[oaicite:1]{index=1}  
    - Selection of numerical flux (e.g., `ChandrashekarFlux`) based on runtime
      configuration.
    - Construction of `DGSEMOperator` with arguments:
      - FE spaces (`vfes`, `fes0`, etc.)  
      - Parallel mesh (`pmesh`)  
      - Blending coefficient fields (e.g., `alpha`)  
      - Gradient fields (`dudx`, `dudy`, `dudz`)  
      - DGSEM integrator object  
      - Indicator/limiter (e.g., Persson–Peraire)  
      - Physics constants (gamma)
    - This is where `DGSEMNonlinearForm` is created and owned by the operator.
  - Why it matters:
    - This call site defines the lifetime and ownership relationships between
      MFEM spaces, the DGSEM operator, and the nonlinear form.
    - Any future GasModel or LTE support will likely be passed into the
      operator here (instead of raw PhysicsConstants usage). :contentReference[oaicite:2]{index=2}  
    - GPU/PA enablement will need to extend DGSEMOperator and its integrator
      construction to set up device-ready data.

- **Boundary condition wiring**
  - File: `Simulation.cpp`  
  - What to look for:
    - Loops over boundary segments reading BC types from the runtime config.  
    - Creation of boundary integrator instances such as:
      - Symmetry and axisymmetric BCs  
      - No-slip adiabatic walls (including heat and velocity boundary
        conditions from ConditionFactory) :contentReference[oaicite:3]{index=3}  
    - Calls to `NS->AddBdrFaceIntegrator(...)` with integrator pointers and
      boundary marker arrays.
  - Why it matters:
    - This is the central registry for which BC classes are in play and how
      they map to mesh boundary attributes.
    - StateView and GasModel must eventually be wired into these integrators
      so BCs use the same layout and EOS logic as the rest of the solver.

- **Time integrator setup**
  - File: `Simulation.cpp`  
  - What to look for:
    - Construction of an MFEM ODE solver (e.g., Forward Euler or Runge–Kutta)
      based on runtime configuration.
    - The call:
      - `NS->SetTime(t);`  
      - `ode_solver->Init(*NS);`  
      which registers DGSEMOperator as the `TimeDependentOperator`. :contentReference[oaicite:4]{index=4}  
    - The main time-stepping loop, where:
      - A time step dt is chosen (possibly via CFL calculation using mesh
        element sizes and max characteristic speeds from NS).  
      - `ode_solver->Step(*sol, t, dt_real);` advances the solution.
  - Why it matters:
    - This is the only place where the ODE solver sees the PDE; it only knows
      about `sol`, `t`, and `NS`.  
    - Any changes to how the state is stored (e.g., device memory vs host, or
      PA vs full assembly) must be reflected in DGSEMOperator and its
      integration with MFEM here, not in the ODE solver.

Together, these locations describe how Prandtl maps from input/runtime state
to MFEM objects, to the DGSEM operator, and finally into time integration.
They are the primary “wiring harness” you will revisit when introducing
StateView, GasModel, and GPU/PA execution paths.




### 4.3 Checklist: Locations of Raw State Indexing (rho, rhoU, rhoV, E)

The following list identifies the major patterns and code regions where the
conserved state is accessed through manual indexing. These locations are the
primary targets for replacement with the new StateView API.

This checklist is not exhaustive, but it captures all high-impact sites based
on the current code structure.

---

#### Pattern 1: Direct use of sol->GetData()
- File: `Simulation.cpp`  
- Typical usage:
  - `real_t* sol_state = sol->GetData();`  
  - Access via `sol_state[eq*num_dofs_scalar + i]`  
- Consumers:
  - Debug printing  
  - Derivation of rho, momentum components, energies  
  - Velocity and pressure computation for visualization  
- Why it matters:
  - This code explicitly documents the assumed equation-blocked layout and
    is tightly coupled to the indexing scheme.  
  - All of these accesses must be replaced by StateView.

---

#### Pattern 2: Manual component extraction inside DGSEMOperator
- File: `src/Operators/DGSEMOperator.cpp`  
- Typical usage:
  - Access to U, or Ustate (axisymmetric path)  
  - Direct extraction of conserved components for:
    - Entropy operations  
    - FV blending  
    - Primitive variable computations  
- Why it matters:
  - DGSEMOperator::Mult is the first consumer of the state during RHS evaluation.  
  - All state layout assumptions must be centralized via StateView here.

---

#### Pattern 3: Element DOF extraction in DGSEMNonlinearForm
- File: `src/Operators/NonlinearForm/DGSEMNonlinearForm.cpp`  
- Typical usage patterns:
  - Gathering element DOFs into local arrays  
  - Gathering face-neighbor DOFs  
  - Computing primitive variables (velocity components, kinetic energy)  
  - Using density and momentum directly for flux evaluation  
- Why it matters:
  - This is the performance-critical core of the DGSEM kernel.  
  - Raw indexing must be replaced by structured accessors suitable for PA/GPU.  
  - Device execution requires eliminating indirect indexing and branching.

---

#### Pattern 4: Numerical flux classes
- Files: `src/Fluxes/*` (e.g., Navier–Stokes flux, Chandrashekar flux)  
- Typical usage:
  - Extracting rho, velocity, pressure, enthalpy from the local U vector  
  - Often uses expressions like:
    - rho = U[0]  
    - rho_u = U[1]  
    - rho_v = U[2]  
    - E = U[3]  
- Why it matters:
  - Flux routines operate per quadrature point and must be GPU-friendly.  
  - These routines should ideally operate on small local views (e.g., a 
    per-element or per-face micro-view provided by StateView).  
  - Removing raw indexing here will greatly improve clarity and safety.

---

#### Pattern 5: Boundary integrators
- Files: `src/Operators/Boundary/*`  
- Typical usage:
  - Extracting primitive variables from U_int for:
    - No-slip walls  
    - Symmetry/axisymmetric BCs  
    - Adiabatic or isothermal boundary conditions  
  - Manual computation of velocities, temperature, pressure  
- Why it matters:
  - BC code often duplicates state access logic that should be unified under 
    StateView.  
  - EOS usage here is inconsistent and will need GasModel integration later.

---

#### Pattern 6: Postprocessing (derived fields)
- File: `Simulation.cpp`  
- Typical usage:
  - Loops that define fields `u`, `v`, `w`, `p`, etc., in terms of  
    rho, momentum components, and energy:  
    - mom(i) / rho(i)  
    - p(i) = gammaM1 * (E(i) - 0.5*rho(i)*V_sq)  
- Why it matters:
  - These computations make the state layout explicit yet duplicated.  
  - StateView should become the sole interface for accessing the conserved
    variables, and postprocessing should rely on it just as the solver does.

---

This checklist should be used as an actionable guide during the StateView
implementation. Every bullet above corresponds to a region of code that must
be reviewed and updated so that no raw state indexing remains.


### 4.4 EOS / Physics Touchpoints (Preparation for GasModel Integration)

This subsection identifies where EOS- and physics-related logic currently
enters the code. These locations are the primary candidates for refactoring
toward a unified GasModel interface, as described in the development plan. :contentReference[oaicite:0]{index=0}  

The goal is to replace scattered, ideal-gas-specific logic with a clean,
encapsulated API that can later support LTE and more complex models without
touching DGSEM internals.

---

#### Touchpoint 1: PhysicsConstants (global ideal-gas parameters)

- File: `Simulation.cpp`  
- What it does:
  - Constructs `PhysicsConstants` using runtime parameters:
    - gamma  
    - Prandtl number  
    - R_gas  
    - Viscosity parameters (mu, mu0, mu_bulk, Ts, T0) :contentReference[oaicite:1]{index=1}  
- Why it matters:
  - PhysicsConstants is the current “poor man’s EOS,” pushed throughout the
    solver wherever thermodynamic quantities are needed.
  - GasModel will eventually supersede this pattern; PhysicsConstants can be
    reduced to a container of model configuration parameters, not the EOS
    interface itself.

---

#### Touchpoint 2: Navier–Stokes flux construction

- Files: `Simulation.cpp`, `src/Fluxes/*`  
- What to look for:
  - Construction of `NavierStokesFlux` with PhysicsConstants (gamma, Pr,
    mu, R_gas, etc.). :contentReference[oaicite:2]{index=2}  
  - Use of these parameters to compute inviscid and viscous fluxes, sound
    speed, or other derived quantities.
- Why it matters:
  - Flux routines implicitly assume an ideal-gas EOS and a particular
    relationship between conserved and primitive variables.
  - GasModel should become the authoritative source for:
    - Pressure
    - Temperature
    - Sound speed
    - Enthalpy and related closures

---

#### Touchpoint 3: Numerical fluxes (Riemann solvers)

- Files: `src/Fluxes/*` (e.g., Chandrashekar flux)  
- What to look for:
  - Use of gamma and other ideal-gas assumptions inside numerical flux
    evaluation:
    - Reconstruction of primitive variables from U  
    - Computation of Roe-type averages, wave speeds, etc.
- Why it matters:
  - Numerical fluxes currently embed EOS details directly.
  - After GasModel integration, these should:
    - Accept primitive/conserved data in a layout-independent way (via
      StateView or thin local views).  
    - Query EOS properties through GasModel instead of recomputing them
      ad hoc.

---

#### Touchpoint 4: Entropy-related and primitive-variable computations

- Files: `src/Operators/DGSEMOperator.cpp`, `src/Operators/NonlinearForm/DGSEMNonlinearForm.cpp`  
- What to look for:
  - Computation of:
    - Entropy or entropy variables  
    - Pressure, temperature, and internal energy from U  
  - Direct use of gamma or other ideal-gas constants in these formulas.
- Why it matters:
  - These are precisely the calculations that will diverge from simple
    ideal-gas behavior under LTE or more advanced models.
  - GasModel should eventually host these transformations:
    - cons → prim and prim → cons  
    - EOS closures needed by entropy/viscous terms

---

#### Touchpoint 5: Boundary conditions with thermodynamic logic

- Files: `src/Operators/Boundary/*`  
- What to look for:
  - BCs that:
    - Enforce isothermal walls  
    - Use temperature or pressure directly  
    - Depend on gamma or R_gas to build ghost states
- Why it matters:
  - BC code currently mixes:
    - State layout assumptions  
    - EOS-specific formulas  
    - Geometry and boundary logic  
  - Under GasModel, BCs should:
    - Use StateView for layout  
    - Call GasModel for thermodynamic relations  
    - Focus only on spatial/boundary semantics (e.g., no-slip, adiabatic)

---

#### Touchpoint 6: Mutation++ and future LTE/NLTE models

- Current status:
  - Mutation++ is not yet wired through Prandtl’s MFEM-based solver, but the
    development plan anticipates its use for LTE (and later NLTE) modeling. :contentReference[oaicite:3]{index=3}  
- Constraints:
  - Mutation++ cannot be called from GPU kernels.  
  - Any Mutation++ usage must remain host-side or be converted into tabulated
    data passed to device kernels.
- Why it matters:
  - GasModel must be designed with this constraint in mind:
    - Clearly separate host-only EOS evaluation from device-side usage.  
    - Provide interfaces that can be backed by:
      - Analytical ideal-gas models  
      - Mutation++-driven LTE tables  
      - Future mixture models

---

In summary, all EOS/physics logic that currently uses PhysicsConstants or
hard-coded ideal-gas formulas should eventually route through GasModel.
StateView will first stabilize state layout; GasModel will then provide a
single, GPU-aware interface for all thermodynamic operations without forcing
changes to DGSEMOperator or DGSEMNonlinearForm.
