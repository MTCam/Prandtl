# Phase 1

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

