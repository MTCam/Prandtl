### Goals
- Align Prandtl abstractions/layers/constructs with MFEM  
- Prepare DGSEM layer for GPU execution  

### Steps
- **Development process**
  - [x] Create Issue for Track B Phase 1  
  - [x] Create top-level branch for device-readiness  
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
