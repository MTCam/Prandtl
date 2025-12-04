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
- [x] Document the development plan
- [x] Discuss working agreement, branching model, PR coordination  

### Deliverables
- [x] Merge-ready PR for Phase 0  
- [x] Integration branch for combined Track A+B testing  
- [x] Initial version of development plan text  
- [x] Regression tests to protect correctness  

### Outcomes
- Team readiness for coordinated multi-track development  
- Stable baseline to protect correctness during major architectural changes  
