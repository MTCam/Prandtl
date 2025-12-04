# Prandtl Modernization Plan: GPU Enablement + EOS/LTE Capability

## Overview
This is a draft development plan for "modernizing" Prandtl. The current
development plan follows two main tracks corresponding to parallel efforts
to extend Prandtl with new capabilities. Extra care must be taken to
coordinate these development efforts, lest we make a giant mess of the code,
and our lives.

The development tracks are:

- **Track A:** Add LTE Capability to Prandtl  
- **Track B:** Add GPU Execution Capability to Prandtl

The planned steps for carrying out this work are structured into phases.
Each phase may contain work for zero, one, or both tracks. Where no track
is listed, the phase applies to both tracks simultaneously.

Coordination points are tagged `[c:A]`, `[c:1B]`, etc.  
Discussion/question points are tagged `[Q1]`, `[Q2]`, etc, and link to the
[Questions & Discussions Section](#questions-and-discussions)
---

# Phase 0 — Development Readiness (Both Tracks)

## Branch management plan/overview
- Tracks A & B will each maintain separate long-lived top-level feature development branches
- One integration branch will be the shared upstream point
- Developments needed by both tracks should merge to the integration branch, then flow down into the Track-specific developments
- None of this development should merge to main until integration is stabilized and tests are green

### Goals
- Prandtl is not broken by development changes  
- Ensure coordination between physics/DGSEM/GPU developers  
- Establish shared expectations about branches, PR workflow, and testing  
- Craft and document development plan

### Steps
- **Development process**
  - [x] Make an Issue to track Phase 0 readiness work
  - [x] Make a top-level branch to track Phase 0 development
  - [x] Make a top-level branch to encapsulate LTE (Track A) development
  - [x] Make a top-level branch to encapsulate GPU (Track B) development 
  - [x] Make an integration branch to test Track A/B integration
  - [x] Create a (draft) PR that closes the readiness Issue

- [x] **Prepare baseline tests (Euler, ideal gas)**
  - [x] 1D Sod shock tube
  - [x] 2D Euler test
  - [x] CNS Problem with nontrivial BCs
  - [x] Record conserved quantities and expected profiles
  
- [x] Add regression testing infrastructure
- [x] Document the development plan (**this document**)
- [x] Discuss working agreement, branching model, PR coordination  

### Deliverables
- [x] Merge-ready PR for Phase 0  
- [x] Integration branch for combined Track A+B testing  
- [x] Initial version of *this* development plan text  
- [x] Regression tests to protect correctness  

### Outcomes
- Team readiness for coordinated multi-track development  
- Stable baseline to protect correctness during major architectural changes  

---

# Phase 1

---

## Track A — EOS/Gas-Model Plan & Design

### Goals
- Design an abstraction for encapsulating EOS/Transport/Physics logic  
- Prepare for LTE capability by removing scattered ideal-gas assumptions  
- Craft a (markdown) Plan/Design document

### Steps
- **Development process**
  - [ ] Create an Issue for Track A Phase 1  
  - [ ] Create a top-level branch for EOS/Gas-Model Planning   
  - [ ] Create a (draft) PR that closes the Issue  

- **Create a "thermo usage inventory"**
  - [ ] List all thermal-state-related `compute_*` routines  
  - [ ] Record inputs (conserved or primitive)  
  - [ ] Record call sites where each is used (flux, BC, sensors, parabolic terms, etc.)

- **Inventory assumptions about state layout** [`[Q1]`](#q1--state-layout-stability-high-priority-before-development)
  - rho, momentum, energy ordering  
  - Future scalar transport and mixture concerns `[c:1B]`  
  - [ ] **Determine/document state interface requirements**  
  - [ ] **Determine/document needs for persistent auxiliary state data** [`[Q2]`](#q2--persistent-state-requirements)

- **Identify MFEM Euler helper usage**
  - Euler/inviscid fluxes (for example) 
  - These must be replaced by Prandtl-native kernels to support mixtures/scalars `[c:1B]`

- [ ] Prepare testing extensions as needed
  - EOS unit testing likely needed

### Deliverables
- [ ] Merge-ready PR for Track A Phase 1  
- [ ] Markdown/table summarizing thermo-usage and state-layout assumptions  
- [ ] Documented Track A plan for:
  - Adding EOS enapsulation
  - Adding GasModel construct

### Outcomes
- Clarity on what must be encapsulated before EOS/LTE work  
- Shared understanding of state-layout constraints  
- Regression tests ensure safe development
- Unit tests to ensure correctness, definitiion-of-done

---

## Track B — Device Development Readiness

### Goals
- Align Prandtl abstractions/layers/constructs with MFEM  
- Prepare DGSEM layer for GPU execution  

### Steps
- **Development process**
  - [ ] Create Issue for Track B Phase 1  
  - [ ] Create top-level branch for device-readiness  
  - [ ] Draft PR closing the Issue  

- **Centralize and abstract state layout** [`[Q1]`](#q1--state-layout-stability-high-priority-before-development)
  - [ ] Add unit tests for state
  - [ ] Add accessors for all state components  
  - [ ] Ensure layout supports future scalar transport `[c:A]`  
  - [ ] Plan for persistent primitive/derived states [`[Q2]`](#q2--persistent-state-requirements)

- [ ] **Make `DGSEMOperator` a thin MFEM `TimeDependentOperator`**
  - Axisymmetric weighting  
  - FV blending coefficient  
  - Entropy conversion plumbing [`[Q3]`](#q3--entropy-conversion-plumbing)  
  - Calls `DGSEMNonlinearForm` for spatial discretization  

- [ ] **Make `DGSEMNonlinearForm` the exclusive home of PDE spatial discretization**
  - Volume convective terms  
  - Viscous/parabolic terms  
  - Interior face fluxes  
  - Boundary faces and BC integrators  
  - Future LTE source terms `[c:A]`  

- [ ] **Refactor `DGSEMNonlinearForm::Mult` into logical blocks** *(no math changes)*  

- [ ] **Device-sanitize all hot loops**
  - No new/delete, malloc, STL, or iostream  
  - No virtual calls in quadrature loops  
  - No Mutation++ or host-only calls in device paths `[c:A]`  

- [ ] **Align BCs and integrators**
  - Ensure `AddBdrFaceIntegrator` forwards cleanly  
  - BCs must use GasModel, not ad-hoc EOS logic `[c:A]`

### Deliverables
- [ ] Merge-ready PR for Track B Phase 1  
- [ ] State accessors
- [ ] reorganized DGSEMOperator
- [ ] structured DGSEMNonlinearForm  

### Outcomes
- MFEM-style operator structure suitable for GPU backend  
- Centralized state definition ready for scalar transport and LTE  

---

# Phase 2

---

## Track A — EOS/Gas-Model Development (Ideal Gas First)

### Goals
- Encapsulate ideal-gas EOS into single abstraction  
- Replace scattered thermodynamic calls with GasModel usage  

### Steps
- **Development process**
  - Make Issue  
  - Create development branch  
  - Draft PR  

- **Define minimal GasModel interface**
  - Cons↔Prim transforms  
  - Pressure, temperature, sound speed  
  - **NOTE:** Entropy conversions MAY live here (decision deferred) [`[Q3]`](#q3--entropy-conversion-plumbing)

- Implement IdealGas EOS (must match existing behavior)  
- Add EOS unit tests  
- Introduce legacy shims (`compute_* → gas_model.*`)  
- Switch BCs/fluxes to use gas_model.* `[c:1B]`  

### Deliverables
- Merge-ready PR with IdealGas GasModel  
- GasModel/EOS classes, unit tests, shims  

### Outcomes
- Single source of truth for thermodynamics  
- Ready for LTE work without physics changes  

---

## Track B — Add GPU / Partial Assembly Capability

### Goals
- Implement MFEM-style PA for DGSEM  
- Enable Prandtl to run DGSEM kernels on GPUs  

### Steps
- **Development process**
  - Make Issue, branch, PR  

- For each DGSEM block:
  - Implement PA device version (AssemblePA, AddMultPA)  
  - Keep CPU fallback  

- Keep DGSEMOperator thin  
- Validate GPU performance and correctness  

### Deliverables
- Merge-ready PR enabling GPU execution  
- Performance results  

### Outcomes
- Ideal-gas Prandtl GPU-ready  
- PA integrators prepared for LTE/NLTE/scalars  

---

# Phase 3

---

## Track A — Add LTE EOS Capability Behind GasModel

### Goals
- Extend GasModel for LTE thermodynamics  
- Add LTE internal variables & closures without touching DGSEM internals  

### Steps
- **Development process**
  - Make Issue, branch, PR  

- Extend GasModel/EOS:
  - LTE internal modes, electronic states  
  - LTE closure for p, T, h, c_s  
  - LTE source terms  

- Handle Mutation++ carefully:
  - **Cannot be called from GPU**  
  - Must not appear in device paths  
  - Host-side or tabulated data fed to kernels [`[Q4]`](#q4--mutation-and-gpu)  

- Add LTE examples/tests  

### Deliverables
- Merge-ready PR with LTE support  
- LTE examples  

### Outcomes
- LTE gas supported cleanly  
- GPU readiness maintained  

---

# Phase 4 — Planning for NLTE, Chemistry, Scalar Transport

### Draft Plan
- Add scalar transport (passive or species)  
- Update state layout/accessors for more components  
- Add volume/face/diffusive scalar fluxes  
- Manufactured tests  
- Add mixture EOS models  
- Plan GPU-friendly mixture representations  

### Outcomes
- Ready for full thermo-chemical flows  
- Architecture future-proof  

---

# Key Coordination GOTCHAs (GPU + LTE Work)

- State layout stabilized early (future scalars) [`[Q1]`](#q1--state-layout-stability-high-priority-before-development)  
- GasModel API must stabilize before GPU kernels depend on it  
- No raw indexing—use state accessors only  
- No Mutation++ in GPU kernels [`[Q4]`](#q4--mutation-and-gpu)  
- No virtual dispatch in device loops  
- DGSEMOperator stays thin  
- Consider persistent primitive/derived state [`[Q2]`](#q2--persistent-state-requirements)  
- CI stays green  

---

# Questions and Discussions


### Q1 — State Layout Stability (high priority, before development)
- Must eliminate hard-coded direct access of state
- Plan: develop lightweight state wrapper/accessor class
- Must plan for future scalars and mixture variables  
- Layout should not break newly written kernels later  

### Q2 — Persistent State Requirements
- Do we need to track primitive/derived quantities persistently?  
- Some primitive or derived quantities may need persistent storage  
- Implications for:
  - device kernels  
  - GasModel interface design  
  - LTE (and later scalar transport) state encapsulation  

### Q3 — Entropy Conversion Plumbing
- Entropy conversions **may** belong in the GasModel  
- Decision deferred  

### Q4 — Mutation++ and GPU
- Cannot be called from GPU kernels  
- Host-only or tabulated data must be used  

---

# End of Plan
