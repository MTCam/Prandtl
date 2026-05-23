#include "DGSEMOperator.hpp"

namespace Prandtl
{
  DGSEMOperator::DGSEMOperator(std::shared_ptr<ParFiniteElementSpace> vfes_,
                               std::shared_ptr<ParFiniteElementSpace> fes0_,
                               std::shared_ptr<ParMesh> pmesh_,
                               std::shared_ptr<ParGridFunction> eta_,
                               std::shared_ptr<ParGridFunction> alpha_,
                               std::vector<std::shared_ptr<ParGridFunction> > &grad_u_,
                               std::unique_ptr<DGSEMIntegrator> integrator_,
                               std::unique_ptr<Indicator> indicator_,
                               const ActiveGasModel &gasModel_,
                               const HostThermoTables &thermoTables_,
                               std::shared_ptr<ParGridFunction> r_gf_,
                               const real_t alpha_max, const real_t alpha_min)
  : TimeDependentOperator(vfes_->GetTrueVSize()),
    vfes(vfes_), fes0(fes0_), pmesh(pmesh_),
    eta(eta_), alpha(alpha_), grad_u(grad_u_),
    integrator(std::move(integrator_)), indicator(std::move(indicator_)),
    gasModel(gasModel_),
    thermoTables(thermoTables_),
    num_equations(vfes->GetVDim()), dim(pmesh->SpaceDimension()),
    order(vfes->GetElementOrder(0)), num_elements(pmesh->GetNE()),
    Ndofs(vfes->GetFE(0)->GetDof()),
    modalThreshold(0.5 * std::pow(10.0, -1.8 * std::pow(order, 0.25))),
    r_gf(r_gf_), alpha_max(alpha_max), alpha_min(alpha_min),
    num_dofs_scalar(vfes_->GetTrueVSize()/vfes_->GetVDim())
#ifdef AXISYMMETRIC
  , U(vfes->GetTrueVSize())
#endif
  {
    nonlinearForm.reset(new DGSEMNonlinearForm(vfes.get()));
    
    nonlinearForm->AddDomainIntegrator(integrator.get());
    nonlinearForm->AddInteriorFaceIntegrator(integrator.get());
    
    std::vector<BdrFaceIntegrator*>::iterator it1 = bfnfi.begin();
    std::vector<Array<int>>::iterator it2 = bdr_marker.begin();
    
    for (; it1 != bfnfi.end() && it2 != bdr_marker.end(); ++it1, ++it2)
      {
        nonlinearForm->AddBdrFaceIntegrator(*it1, *it2);
      }
    nonlinearForm->UseExternalIntegrators();
    
#ifdef PARABOLIC
    global_entropy.SetSize(vfes->GetVSize());
#endif

    diag0.mass = 0.0;
    diag0.ke = 0.0;
    diag0.en = 0.0;

  }

  void DGSEMOperator::Finalize(real_t time)
  {    
    this->SetTime(time);
    GetOperatorCache(vfes.get(), &operator_cache);
    AssembleBoundaryFaceGeometryTerms(vfes.get(), bdr_marker, &operator_cache);
#ifdef SUBCELL_FV_BLENDING
    ComputeSubcellMetrics(vfes.get(), &operator_cache);
#endif
    // Important that gasModel is POD for host<->device
    operator_cache.gas = gasModel;
    operator_cache.thermoTables = thermoTables;
    operator_cache.alpha = alpha;
    operator_cache.bc_descriptors = bc_descriptors;
    operator_cache.bc_scalar_data = bc_scalar_data;
    operator_cache.bc_vector_data = bc_vector_data;

    GetDeviceCache(operator_cache, device_cache);

    nonlinearForm->SetOperatorCache(&operator_cache);
    integrator->SetOperatorCache(&operator_cache); 
  }


  DGSEMOperator::~DGSEMOperator()
  {
    for (auto ptr : bfnfi)
      {
        delete ptr;
      }
  }

#ifdef SUBCELL_FV_BLENDING  
  void DGSEMOperator::ComputeBlendingCoefficient(const Vector &x) const
  {
    indicator->CheckSmoothness(x);
    for (int el = 0; el < num_elements; el++)
      {
        fes0->GetElementDofs(el, ind_indx);
        eta->GetSubVector(ind_indx, ind_dof);
        alpha_dof = 1.0 / (1.0 + std::exp(-sharpness_fac * (ind_dof(0) - modalThreshold) / modalThreshold));
        if (alpha_dof < alpha_min)
          {
            alpha_dof = 0.0;
          }
        else if (alpha_dof > (1.0 - alpha_min))
          {
            alpha_dof = 1.0;
          }
        alpha_dof = std::min(alpha_dof, alpha_max);
        alpha->SetSubVector(ind_indx, alpha_dof);
      }
    
  }

  void DGSEMOperator::ComputeIndicatorField(const Vector &u, Vector &indicator_field) const
  {
    ScopedTimer timer("ComputeIndicator_Device");

    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();
    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int dim = dc.dim;
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int Np_x = dc.Np_x;
    const int Np_y = dc.Np_y;
    const int Np_z = dc.Np_z;

    MFEM_ASSERT(nval_restr == ne*ndof*neq, "Unexpected size for volume restriction in indicator calc.");
    const int nval_ind = nval_restr / neq;
    mfem::Vector Ue(nval_restr);
    indicator_field.SetSize(nval_ind);
    Ue.UseDevice();
    indicator_field.UseDevice();

    real_t *ifield_d = indicator_field.Write();

    operator_cache.restr_v->Mult(u, Ue);
    const real_t *Ue_d = Ue.Read();
    const int estride = ndof*neq;
    
    // Inside the FORALL below, executed on device
    mfem::forall(nval_ind, [=] MFEM_HOST_DEVICE (int vind)
    {
      const int e = vind / ndof;
      const int evind = vind - e * ndof;
      const real_t *u_el = Ue_d + e * estride;
      real_t elstate[Prandtl::MAXEQ];
      Kernels::el_gather_state(u_el, ndof, neq, evind, elstate);
      Prandtl::PointStateView S{elstate};
      ifield_d[vind] = dc.gas.pressure(S) * dc.gas.density(S);
    });

  }

  void DGSEMOperator::ComputeBlendingCoefficientFromIndicator(const Vector &indicator_field) const
  {
    {
      ScopedTimer timer("CheckIndicatorSmoothness_Host");
      // This is a HOST-only routine.  Make sure the input is host-readable
      indicator->CheckIndicatorSmoothness(indicator_field);
    }
    ScopedTimer timer("ComputeAlpha_Host");
    real_t *alpha_h = alpha->HostWrite();
    const real_t *eta_h = eta->HostRead();
    for (int el = 0; el < num_elements; el++)
      {
        alpha_dof = 1.0 / (1.0 + std::exp(-sharpness_fac * (eta_h[el] - modalThreshold) / modalThreshold));
        if (alpha_dof < alpha_min)
          {
            alpha_dof = 0.0;
          }
        else if (alpha_dof > (1.0 - alpha_min))
          {
            alpha_dof = 1.0;
          }
        alpha_h[el] = std::min(alpha_dof, alpha_max);
      }
  }
 #endif

  void DGSEMOperator::ComputeIntegralMeasures(const Vector &u, IntegralMeasures &diag) const
  {
    
    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;
    
    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const real_t *qWts_d = dc.elQWgts_d;
    auto gas = dc.gas;
    
 
    mfem::Vector Ue(nval_restr);
    Ue.UseDevice();

    operator_cache.restr_v->Mult(u, Ue);
    const real_t *Ue_d = Ue.Read();
    const int estride = ndof*neq;

    mfem::Vector elMass_integral(ne);
    mfem::Vector elKE_integral(ne);
    mfem::Vector elEnergy_integral(ne);
    mfem::Vector elMaxPressure(ne);
    mfem::Vector elMaxTemperature(ne);
    mfem::Vector elMaxDensity(ne);
    mfem::Vector elMinPressure(ne);
    mfem::Vector elMinTemperature(ne);
    mfem::Vector elMinDensity(ne);

    elMass_integral.UseDevice();
    elKE_integral.UseDevice();
    elEnergy_integral.UseDevice();
    elMaxPressure.UseDevice();
    elMaxTemperature.UseDevice();
    elMaxDensity.UseDevice();
    elMinPressure.UseDevice();
    elMinTemperature.UseDevice();
    elMinDensity.UseDevice();

    real_t *elMass_int_d = elMass_integral.Write();
    real_t *elKE_int_d = elKE_integral.Write();
    real_t *elEnergy_int_d = elEnergy_integral.Write();

    real_t *elPress_max_d = elMaxPressure.Write();
    real_t *elTemp_max_d = elMaxTemperature.Write();
    real_t *elDens_max_d = elMaxDensity.Write();
    real_t *elPress_min_d = elMinPressure.Write();
    real_t *elTemp_min_d = elMinTemperature.Write();
    real_t *elDens_min_d = elMinDensity.Write();

    // Inside the FORALL below, executed on device
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      const real_t *u_el = Ue_d + e * estride;
      const real_t *qWgt = qWts_d + e * ndof;
   
      real_t mass_int = 0.0;
      real_t ke_int = 0.0;
      real_t en_int = 0.0;
      real_t min_dens = 1e32;
      real_t max_dens = 0.0;
      real_t min_temp = 1e32;
      real_t max_temp = 0.0;
      real_t min_press = 1e32;
      real_t max_press = 0.0;
      for(int ep = 0;ep < ndof;ep++){
        real_t elstate[Prandtl::MAXEQ];
        Kernels::el_gather_state(u_el, ndof, neq, ep, elstate);
        Prandtl::PointStateView S{elstate};

        real_t rho = gas.density(S);
        real_t ke = gas.kinetic_energy_density(S);
        real_t rhoE = gas.energy(S); // energy density
        real_t press = gas.pressure(S);
        real_t temper = gas.temperature(S);

        mass_int += rho * qWgt[ep];
        ke_int += ke * qWgt[ep];
        en_int += rhoE * qWgt[ep];

        min_temp = Kernels::rmin(min_temp, temper);
        max_temp = Kernels::rmax(max_temp, temper);
        min_dens = Kernels::rmin(min_dens, rho);
        max_dens = Kernels::rmax(max_dens, rho);
        min_press = Kernels::rmin(min_press, press);
        max_press = Kernels::rmax(max_press, press);
      }

      elMass_int_d[e]   = mass_int;
      elKE_int_d[e]     = ke_int;
      elEnergy_int_d[e] = en_int;
      elPress_max_d[e]  = max_press;
      elPress_min_d[e]  = min_press;
      elDens_max_d[e]   = max_dens;
      elDens_min_d[e]   = min_dens;
      elTemp_min_d[e]   = min_temp;
      elTemp_max_d[e]   = max_temp;

    });

    // diag.mass = mfem::Sum(elMass_integral);
    // diag.ke = mfem::Sum(elKE_integral);
    // diag.en = mfem::Sum(elEnergy_integral);
    diag.mass = 0.0;
    diag.ke   = 0.0;
    diag.en   = 0.0;
    diag.min_press = 1e32;
    diag.max_press = 0.0;
    diag.min_dens = 1e32;
    diag.max_dens = 0.0;
    diag.min_temp = 1e32;
    diag.max_temp = 0.0;
    const real_t *mass_h = elMass_integral.HostRead();
    const real_t *ke_h   = elKE_integral.HostRead();
    const real_t *en_h   = elEnergy_integral.HostRead();
    const real_t *minpress_h = elMinPressure.HostRead();
    const real_t *maxpress_h = elMaxPressure.HostRead();
    const real_t *mindens_h = elMinDensity.HostRead();
    const real_t *maxdens_h = elMaxDensity.HostRead();
    const real_t *mintemp_h = elMinTemperature.HostRead();
    const real_t *maxtemp_h = elMaxTemperature.HostRead();

    for (int e = 0; e < ne; ++e) {
      diag.mass += mass_h[e];
      diag.ke   += ke_h[e];
      diag.en   += en_h[e];
      diag.min_press = Kernels::rmin(diag.min_press, minpress_h[e]);
      diag.max_press = Kernels::rmax(diag.max_press, maxpress_h[e]);
      diag.min_temp = Kernels::rmin(diag.min_temp, mintemp_h[e]);
      diag.max_temp = Kernels::rmax(diag.max_temp, maxtemp_h[e]);
      diag.min_dens = Kernels::rmin(diag.min_dens, mindens_h[e]);
      diag.max_dens = Kernels::rmax(diag.max_dens, maxdens_h[e]);
    }

    real_t sendbuf[3] = {diag.mass, diag.ke, diag.en};
    real_t recvbuf[3] = {0.0, 0.0, 0.0};

    MPI_Allreduce(sendbuf, recvbuf, 3, MPITypeMap<real_t>::mpi_type, MPI_SUM, pmesh->GetComm());

    diag.mass = recvbuf[0];
    diag.ke = recvbuf[1];
    diag.en = recvbuf[2];

    sendbuf[0] = diag.min_press;
    sendbuf[1] = diag.min_temp;
    sendbuf[2] = diag.min_dens;

    MPI_Allreduce(sendbuf, recvbuf, 3, MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());

    diag.min_press = recvbuf[0];
    diag.min_temp = recvbuf[1];
    diag.min_dens = recvbuf[2];
 
    sendbuf[0] = diag.max_press;
    sendbuf[1] = diag.max_temp;
    sendbuf[2] = diag.max_dens;

    MPI_Allreduce(sendbuf, recvbuf, 3, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());

    diag.max_press = recvbuf[0];
    diag.max_temp = recvbuf[1];
    diag.max_dens = recvbuf[2];

    if(diag0.mass == 0.0){
      diag0 = diag;
    }

  }

  // Device version of ComputeGlobalEntropyVector
  void DGSEMOperator::ComputeEntropyState(const Vector &u, Vector &e) const
  {
    ScopedTimer timer("ComputeEntropyState_Device");

    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int npts = ndof * ne;

    MFEM_ASSERT(nval_restr == npts*neq, "Unexpected size in ComputeEntropyState");

    auto gas = dc.gas;

    mfem::Vector restrU(nval_restr);
    restrU.UseDevice();
    
    mfem::Vector restrE(nval_restr);
    restrE.UseDevice();
    real_t *eState_d = restrE.Write();

    e.SetSize(u.Size());
    e.UseDevice();

    operator_cache.restr_v->Mult(u, restrU);
    const real_t *restrU_d = restrU.Read();
    const int estride = ndof*neq;

    // Inside the FORALL below, executed on device
    mfem::forall(npts, [=] MFEM_HOST_DEVICE (int pt)
    {
      const int elno = pt / ndof;
      const int ept = pt % ndof;
      const int eoff = elno * estride;
      const real_t *u_el = restrU_d + eoff;

      real_t elUstate[Prandtl::MAXEQ];
      Kernels::el_gather_state(u_el, ndof, neq, ept, elUstate);
      Prandtl::PointStateView S{elUstate};

      real_t elEState[Prandtl::MAXEQ];
      PointStateViewRW E{elEState};
      gas.entropy_state(S, E);
      real_t *e_el = eState_d + eoff;
      Kernels::el_scatter_assign(elEState, ndof, neq, ept, 1.0, e_el);
    });

    operator_cache.restr_v->MultTranspose(restrE, e);

  }

  void DGSEMOperator::ComputeGlobalEntropyVector(const Vector &u, Vector &global_entropy) const
{
  ScopedTimer timer("ComputeEntropyState_Host");
    DenseMatrix ent_mat(Ndofs, num_equations);
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);
        Conserv2Entropy(gasModel, vdof_mat, ent_mat);
        global_entropy.SetSubVector(vdof_indices, ent_mat.GetData());
    }
}

void DGSEMOperator::ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx) const
{
  ScopedTimer timer("GradEnt2GradPrim_Host");
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);

        dudx.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat);
        dudx.SetSubVector(vdof_indices, grad_mat.GetData());
    }
}

// This routine replaces the gradient of the entropy stored in gradState with the gradient
// of the primitive variables so that all the data in gradState is replaced.
void DGSEMOperator::ComputeGradPrimFromGradEntropy(const Vector &u, std::vector<mfem::Vector *> &gradState) const
{
  ScopedTimer timer("GradEntropyToGradPrim_Device");
    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();
    
    // Copy the device cache so that it is not member data
    auto dc = device_cache;
    
    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int npts = ndof * ne;

    MFEM_ASSERT(nval_restr == npts*neq, "Unexpected size in ComputeEntropyState");
    
    auto gas = dc.gas;
    
    mfem::Vector restr_state(nval_restr);
    restr_state.UseDevice();
    
    operator_cache.restr_v->Mult(u, restr_state);
    const real_t *restr_state_d = restr_state.Read();
    const int estride = ndof*neq;

    mfem::Vector restr_grad_prim_dir(nval_restr);
    restr_grad_prim_dir.UseDevice();

  for(int idim = 0;idim < dim;idim++){

    mfem::Vector &grad_state_dir(*gradState[idim]);
    operator_cache.restr_v->Mult(grad_state_dir, restr_grad_prim_dir);
    real_t *grad_prim_dir_d = restr_grad_prim_dir.Write();

    // Inside the FORALL below, executed on device
    mfem::forall(npts, [=] MFEM_HOST_DEVICE (int pt)
    {
      const int e = pt / ndof;
      const int ept = pt % ndof;
      const int eoff = e * estride;
      const real_t *u_el = restr_state_d + eoff;
      real_t *grad_prim_el = grad_prim_dir_d + eoff;

      real_t el_U[Prandtl::MAXEQ];
      Kernels::el_gather_state(u_el, ndof, neq, ept, el_U);
      Prandtl::PointStateView CV{el_U};

      real_t el_gradS[Prandtl::MAXEQ];
      Kernels::el_gather_state(grad_prim_el, ndof, neq, ept, el_gradS);
      Prandtl::PointStateView dS{el_gradS};
      
      real_t el_gradP[Prandtl::MAXEQ];
      PointStateViewRW dP{el_gradP};
      gas.grad_entropy_to_grad_prim(CV, dS, dP);

      Kernels::el_scatter_assign(el_gradP, ndof, neq, ept, 1.0, grad_prim_el);

    });

    operator_cache.restr_v->MultTranspose(restr_grad_prim_dir, grad_state_dir);

  }

}

void DGSEMOperator::ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy) const
{
  ScopedTimer timer("GradEnt2GradPrim_Host");
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);

        dudx.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat1(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat1);
        dudx.SetSubVector(vdof_indices, grad_mat1.GetData());

        dudy.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat2(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat2);
        dudy.SetSubVector(vdof_indices, grad_mat2.GetData());    
        
    }
}

void DGSEMOperator::ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const
{
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);

        dudx.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat1(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat1);
        dudx.SetSubVector(vdof_indices, grad_mat1.GetData());

        dudy.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat2(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat2);
        dudy.SetSubVector(vdof_indices, grad_mat2.GetData());

        dudz.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat3(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat3);
        dudz.SetSubVector(vdof_indices, grad_mat3.GetData());      
    }
}

#ifdef AXISYMMETRIC

    void DGSEMOperator::RecoverStateFromWeighted(const Vector &rU, Vector &U) const
    {
        U.SetSize(rU.Size());

        const real_t tiny = 1e-14;
        const real_t cap_mult = 10.0;
        const real_t z_tol = 1e-12;
        const real_t tiny_detJ = 1e-12;
        const real_t theta_lim = 1.5;
        const real_t rho_floor = rho_floor_abs;
        const real_t p_floor = p_floor_abs;
        long long calls = 0;
        
        enum class AxisReconMode { highOrder_shape, lowOrder_ray1, lowOrder_ray2, lowOrder_copy };
        
        // helper functions
        auto sign = [](real_t a) -> real_t { return (a > 0) - (a < 0); };

        auto minmod = [&](real_t a, real_t b) -> real_t
        {
            if (a * b <= 0.0) { return 0.0; }
            return sign(a) * std::min(std::abs(a), std::abs(b));
        };

        auto clamp = [&](real_t x, real_t m) -> real_t
        {
            if (!isfinite(x)) { return 0.0; }
            const real_t ax = std::abs(x);
            return (ax <= m) ? x : sign(x) * m;
        };

        auto is_dof_on_axis = [&](int dof_id) -> bool 
        {
            const int* beg = axis_idx.GetData();
            const int* end = beg + axis_idx.Size();
            return std::binary_search(beg, end, dof_id);
        };

        Array<int> vdofs;

        for (int e = 0; e < vfes->GetNE(); e++)
        {
            vfes->GetElementVDofs(e, vdofs);

            bool troubled = false;

            if (alpha && fes0)
            {
                Array<int> ind;
                fes0->GetElementDofs(e, ind);
                Vector a_loc;
                alpha->GetSubVector(ind, a_loc);
                troubled = (a_loc.Size() > 0) ? (a_loc(0) > 0.5 * alpha_max) : false;
            }

            Vector rU_e, r_e;
            rU.GetSubVector(vdofs, rU_e);
            MFEM_ASSERT(r_gf != nullptr, "r_gf is null");
            r_gf->GetSubVector(vdofs, r_e);

            Vector U_e(rU_e.Size());

            const DenseMatrix rU_e_mat(rU_e.GetData(), Ndofs, num_equations);
            DenseMatrix U_e_mat(U_e.GetData(), Ndofs, num_equations);
            const Vector r_e_vec(r_e.GetData(), Ndofs);

            const FiniteElement& fe = *vfes->GetFE(e);
            ElementTransformation& Tr = *vfes->GetElementTransformation(e);
            const IntegrationRule& nodes = fe.GetNodes();

            troubled = troubled || low_order_axis;

            for (int ld = 0; ld < Ndofs; ld++)
            {
                const real_t r = r_e_vec(ld);
                const int true_dof_id = vdofs[ld];
                const bool is_axis_node = is_dof_on_axis(true_dof_id);

                // true axis nodes and near axis nodes

                if (!is_axis_node && r > 0.0) // off axis nodes
                {
                    for (int eq = 0; eq < num_equations; eq++)
                    {
                        U_e_mat(ld, eq) = rU_e_mat(ld, eq) / r;
                    }
                    continue;
                }

                // on axis nodes
                calls++;
                AxisReconMode mode = AxisReconMode::highOrder_shape;

                // HIGH ORDER construction of U(0) = d/dr(rU)|(r=0)
                // df/dr = (J^-T * grad_ref f)_r
                const IntegrationPoint &ip = nodes.IntPoint(ld);
                Tr.SetIntPoint(&ip);

                DenseMatrix dshape(Ndofs, dim);
                fe.CalcDShape(ip, dshape);

                Vector X_axis(dim);
                Tr.Transform(ip, X_axis);
                const real_t z_axis = X_axis(0);

                Vector rrho_e(Ndofs), rmz_e(Ndofs), rE_e(Ndofs);

                for (int i = 0; i < Ndofs; i++)
                {
                    rrho_e(i) = rU_e_mat(i, 0);
                    rmz_e(i)  = rU_e_mat(i, 1);
                    rE_e(i)   = rU_e_mat(i, 3);
                }

                // derivative function with fallbacks
                auto d_dr = [&](const Vector& f_e)-> real_t
                {
                    real_t d_dxi = 0.0;
                    real_t d_deta = 0.0;
                    for (int i = 0; i < Ndofs; i++)
                    {
                        const real_t fa = f_e(i);
                        d_dxi += fa * dshape(i, 0);
                        d_deta += fa * dshape(i, 1);
                    }

                    const DenseMatrix &Jinv = Tr.InverseJacobian();
                    DenseMatrix JinvT(Jinv);
                    JinvT.Transpose();

                    Vector grad_ref(dim);
                    grad_ref = 0.0;
                    grad_ref(0) = d_dxi;
                    grad_ref(1) = d_deta;

                    Vector grad_phys(dim);
                    JinvT.Mult(grad_ref, grad_phys);
                    const real_t detJ = Tr.Weight();

                    const real_t df_dr_geom = grad_phys(1);

                    // same-z nearest off axis candidates
                    int j1 = -1, j2 = -1;
                    real_t r1 = infinity(), r2 = infinity();
                    
                    for (int j = 0; j < Ndofs; j++)
                    {
                        const real_t rj = r_e_vec(j);
                        if (rj <= 0.0) { continue; }
                        const IntegrationPoint &ipj = nodes.IntPoint(j);
                        Tr.SetIntPoint(&ipj);
                        Vector Xj(dim);
                        Tr.Transform(ipj, Xj);
                        const real_t zj = Xj(0);
                        if (std::abs(zj - z_axis) <= z_tol)
                        {
                            if (rj < r1)
                            {
                                r2 = r1;
                                j2 = j1;
                                r1 = rj;
                                j1 = j;
                            }
                            else if (rj < r2)
                            {
                                r2 = rj;
                                j2 = j;
                            }
                        }
                    }
                    // if no same z, fallback to globally nearest two off-axis
                    if (j1 < 0)
                    {
                        for (int j = 0; j < Ndofs; j++)
                        {
                            const real_t rj = r_e_vec(j);
                            if (rj > 0.0 && rj < r1)
                            {
                                r2 = r1;
                                j2 = j1;
                                r1 = rj;
                                j1 = j;
                            }
                            else if (rj > 0.0 && rj < r2)
                            {
                                r2 = rj;
                                j2 = j;
                            }
                        }
                    }

                    // cap large gradients
                    const bool have_cap = (j1 >= 0);
                    const real_t f_ref = have_cap ? std::abs(f_e(j1)) : 0.0;
                    const bool ref_too_small = f_ref < 1e-12;
                    const real_t cap = (have_cap && !ref_too_small) ?  (cap_mult * f_ref / std::max(r1, tiny)) : infinity();
                    const bool geom_ok = isfinite(df_dr_geom) && isfinite(detJ) && (std::abs(detJ) > tiny_detJ) && (std::abs(df_dr_geom) <= cap) && !troubled;

                    if (geom_ok)
                    {
                        mode = AxisReconMode::highOrder_shape;
                        return df_dr_geom;
                    }

                    // LOW ORDER fallbacks
                    // 1st order fallback: 2 points least squares along ray + limiter
                    if (j1 >= 0 && j2 >= 0 && isfinite(r1) && isfinite(r2) && r1 > 0.0 && r2 > 0.0)
                    {
                        const real_t f1 = f_e(j1), f2 = f_e(j2);
                        // fitting through the origin: f = mr -> m = (r1 f1 + r2 f2)/(r1^2 + r2^2)
                        real_t m = (r1*f1 + r2*f2) / (std::max(r1*r1 + r2*r2, tiny));
                        const real_t s1 = f1/r1;
                        const real_t s2 = f2/r2;
                        real_t m_limited = minmod(m, minmod(theta_lim * s1, theta_lim * s2));
                        m_limited = clamp(m_limited, cap);

                        mode = AxisReconMode::lowOrder_ray2;
                        return m_limited;
                    }

                    // 1st order fallback: 1 point + limiter
                    if (j1 >= 0 && isfinite(r1) && r1 > 0.0)
                    {
                        const real_t s = f_e(j1)/r1;
                        mode = AxisReconMode::lowOrder_ray1;
                        return clamp(s, cap);
                    }

                    // 0th order trigger
                    mode = AxisReconMode::lowOrder_copy;
                    return 0.0;
                };

                    real_t rho_axis = d_dr(rrho_e);
                    real_t mz_axis  = d_dr(rmz_e);
                    real_t E_axis   = d_dr(rE_e);

                    if (!isfinite(rho_axis) || rho_axis < rho_floor || !isfinite(E_axis))
                    {
                        // find nearest off-axis neighbor
                        real_t r_neighbor = -1.0;
                        int neighbor_ld = -1;

                        for (int nld = 0; nld < Ndofs; nld++)
                        {
                            const real_t rn = r_e_vec(nld);
                            if ( rn > 0.0 && (neighbor_ld == -1 || rn < r_neighbor)) 
                            {
                                r_neighbor = rn;
                                neighbor_ld = nld;
                            }
                        }
                        // if no same z, fallback to globally nearest off-axis
                        if (neighbor_ld != -1)
                        {
                            for (int eq = 0; eq < num_equations; eq++)
                            {
                               U_e_mat(ld, eq) = rU_e_mat(neighbor_ld, eq) / r_neighbor;
                            }
                            U_e_mat(ld, 2) = 0.0;
                            mode = AxisReconMode::lowOrder_copy;
                        }
                        
                        else
                        {
                            // abort
                            MFEM_ABORT("All axis reconstruction fallbacks failed for a node");
                            break;
                        }
                    }
                    else
                    {
                        // high/low order result
                        U_e_mat(ld, 0) = rho_axis;
                        U_e_mat(ld, 1) = mz_axis;
                        U_e_mat(ld, 2) = 0.0; // rho ur = 0
                        U_e_mat(ld, 3) = E_axis;
                    }

                    switch (mode)
                        {
                            case AxisReconMode::highOrder_shape: highOrder_shape_accum++;
                            break;
                            case AxisReconMode::lowOrder_ray1: lowOrder_ray1_accum++;  
                            break;  
                            case AxisReconMode::lowOrder_ray2: lowOrder_ray2_accum++;  
                            break;  
                            case AxisReconMode::lowOrder_copy: lowOrder_copy_accum++;  
                            break;  
                        }
            }   
                        
            U.SetSubVector(vdofs, U_e);

        }
        calls_accum += calls;

    }

    DGSEMOperator::AxisReconStats DGSEMOperator::GetAxisReconStats(bool global) const
    {
        AxisReconStats s {calls_accum, highOrder_shape_accum , lowOrder_ray2_accum, lowOrder_ray1_accum, lowOrder_copy_accum};
        if (!global) return s;

        long long in[5] = { s.calls, s.highOrder_shape, s.lowOrder_ray2, s.lowOrder_ray1, s.lowOrder_copy }, out[5] = {0, 0, 0, 0, 0};
        MPI_Allreduce(in, out, 5, MPI_LONG_LONG, MPI_SUM, pmesh->GetComm());
        return AxisReconStats{ out[0], out[1], out[2], out[3], out[4]};
    }


    void DGSEMOperator::BuildAxisIndexFromMarker()
    {
        axis_idx.SetSize(0);
        if (axis_marker.Size() == 0) { return; }

        ParMesh &pm = *pmesh;
        FiniteElementSpace &fes_scalar = *vfes;

        Array<int> el_dofs;
        const int nbe = pm.GetNBE();

        for (int be = 0; be < nbe; be++)
        {
            const int a = pm.GetBdrAttribute(be);
            if (a <= 0 || a > axis_marker.Size() || axis_marker[a-1] == 0) { continue; }
            
            int el = -1, lf = -1;
            pm.GetBdrElementAdjacentElement(be, el, lf);
            MFEM_ASSERT(el >= 0 && lf >= 0, "Invalid boundary element mapping.");

            fes_scalar.GetElementDofs(el, el_dofs);
            const FiniteElement *fe = fes_scalar.GetFE(el);

            const IntegrationRule &nodes = fe->GetNodes();

            ElementTransformation &Tr = *fes_scalar.GetElementTransformation(el);


            for (int ld = 0; ld < fe->GetDof(); ld++)
            {
                const IntegrationPoint &ip = nodes.IntPoint(ld);
                Vector X(dim);
                Tr.Transform(ip, X);
                const real_t r = (dim > 1) ? X(1) : 0.0;

                if (std::abs(r) <= 1e-14)
                {
                    axis_idx.Append(el_dofs[ld]);
                }
             
            }
        }

        axis_idx.Sort();
        axis_idx.Unique();
    }

    void DGSEMOperator::ZeroAxisRadialMom(Vector &v) const
    {
        const int n = axis_idx.Size();
        
        if (n == 0) { return; }

        const int mom_r = 2;  // (rho, rho*uz. rho*ur, rhoE)
        auto *vr = v.GetData() + mom_r * num_dofs_scalar;

        for (int i = 0; i < n; i++)
        {
            vr[axis_idx[i]] = 0.0;
        }
    }

#endif

void DGSEMOperator::Mult(const Vector &u, Vector &dudt) const
{
  std::string timername("RHS_");
  timername += use_device_path ? "Device" : "Host";
  ScopedTimer timer(timername.c_str());

#ifdef AXISYMMETRIC
  RecoverStateFromWeighted(u, U);
  
  const Vector &Ustate = U;
#else
  const Vector &Ustate = u;
#endif

  
#ifdef SUBCELL_FV_BLENDING
  if (use_device_path)
    {
      ScopedTimer timer("SubcellBlendingStep_Device");
      mfem::Vector indicator_field;
      // Since the CV are on-device, and computing
      // the indicator requires the CV, we compute
      // the indicator on-device and xfer only
      // alpha (the blending coeff) from host/device.
      ComputeIndicatorField(Ustate, indicator_field);
      ComputeBlendingCoefficientFromIndicator(indicator_field);
    } else {
      ScopedTimer timer("SubcellBlendingStep_Host");
      ComputeBlendingCoefficient(Ustate);
  }
#endif

#ifdef PARABOLIC // VISCOUS

  if(use_device_path){
    {
      ScopedTimer timer("Step_Device");
      mfem::Vector entropyState(Ustate.Size());
      ComputeEntropyState(Ustate, entropyState);
      std::vector<mfem::Vector *> gradPrim(dim);
      // grad_u is a vector of parallel grid functions
      // this bit grabs an mfem::Vector ref.
      for(int idim = 0;idim < dim;idim++){
        gradPrim[idim] = &(*grad_u[idim]);
      }
      nonlinearForm->GradOperator(entropyState, gradPrim);
      ComputeGradPrimFromGradEntropy(Ustate, gradPrim);
      max_char_speed = nonlinearForm->MultCNS(Ustate, gradPrim, dudt);
    }
  } else {
    {
      ScopedTimer timer("Step_Host");
      ComputeGlobalEntropyVector(Ustate, global_entropy);
 
      if (dim == 1)
        {
          nonlinearForm->MultLifting(global_entropy, *grad_u[0]);
          ComputeGlobalPrimitiveGradVector(Ustate, *grad_u[0]);
          nonlinearForm->Mult(Ustate, *grad_u[0], dudt);
        }
      else if (dim == 2)
        {
          nonlinearForm->MultLifting(global_entropy, *grad_u[0], *grad_u[1]);
          ComputeGlobalPrimitiveGradVector(Ustate, *grad_u[0], *grad_u[1]);
          nonlinearForm->Mult(Ustate, *grad_u[0], *grad_u[1], dudt);
        }
      else
        {
          nonlinearForm->MultLifting(global_entropy, *grad_u[0], *grad_u[1], *grad_u[2]);
          ComputeGlobalPrimitiveGradVector(Ustate, *grad_u[0], *grad_u[1], *grad_u[2]);
          nonlinearForm->Mult(Ustate, *grad_u[0], *grad_u[1], *grad_u[2], dudt);
        }
      max_char_speed = integrator->GetMaxCharSpeed();
      for (int b = 0; b < bfnfi.size(); b++)
        {
          max_char_speed = std::max(bfnfi[b]->GetMaxCharSpeed(), max_char_speed);
        }
    }
  }

#else   // INVISCID

  if(use_device_path){

    max_char_speed = nonlinearForm->MultEuler(Ustate, dudt);
 
  } else {

    nonlinearForm->Mult(Ustate, dudt);
    max_char_speed = integrator->GetMaxCharSpeed();
    for (int b = 0; b < bfnfi.size(); b++)
      {
        max_char_speed = std::max(bfnfi[b]->GetMaxCharSpeed(), max_char_speed);
      }
 
  }

#endif
  
#ifdef AXISYMMETRIC
  ZeroAxisRadialMom(dudt);
#endif

}

void DGSEMOperator::AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker_)
{
  bfnfi.push_back(bfi);
  bdr_marker.push_back(bdr_marker_);
  nonlinearForm->AddBdrFaceIntegrator(bfi, bdr_marker_);
}

}
