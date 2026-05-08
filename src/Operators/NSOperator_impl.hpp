#include "DGSEMOperator.hpp"

namespace Prandtl
{

#ifdef SUBCELL_FV_BLENDING

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeIndicatorField(const mfem::Vector &u, mfem::Vector &indicator_field) const
  {
    ScopedTimer timer("ComputeIndicator");

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
    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
    if (indicator_field.Size() != nval_ind){
      indicator_field.SetSize(nval_ind);
      indicator_field.UseDevice();
    }

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

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeBlendingCoefficientFromIndicator(const mfem::Vector &indicator_field) const
  {
    {
      ScopedTimer timer("CheckIndicatorSmoothness_Host");
      // This is a HOST-only routine.  Make sure the input is host-readable
      indicator->CheckIndicatorSmoothness(indicator_field);
    }
    ScopedTimer timer("ComputeAlpha_Host");
    real_t *alpha_h = operator_cache.alpha->HostWrite();
    const real_t *eta_h = eta->HostRead();
    for (int el = 0; el < num_elements; el++)
      {
        real_t alpha_dof = 1.0 / (1.0 + std::exp(-sharpness_fac * (eta_h[el] - modalThreshold) / modalThreshold));
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

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeIntegralMeasures(const mfem::Vector &u, IntegralMeasures &diag) const
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

    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
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
  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeEntropyState(const mfem::Vector &u, mfem::Vector &e) const
  {
    ScopedTimer timer("ComputeEntropyState");

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

    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &restrU(operator_cache.uVol);
    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &restrE(operator_cache.uVol);
    if(e.Size() != u.Size()){
      e.SetSize(u.Size());
      e.UseDevice();
    }

    real_t *eState_d = restrE.Write();
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

  // This routine replaces the gradient of the entropy stored in gradState with the gradient
  // of the primitive variables so that all the data in gradState is replaced.
  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeGradPrimFromGradEntropy(const mfem::Vector &u, std::vector<mfem::Vector *> &gradEntropy) const
  {
    ScopedTimer timer("GradEntropyToGradPrim");
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

    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &restr_state(operator_cache.uVol);
    operator_cache.restr_v->Mult(u, restr_state);

    const real_t *restr_state_d = restr_state.Read();
    const int estride = ndof*neq;

    // Leave this temporary for now
    if(operator_cache.volAux.Size() != nval_restr){
      operator_cache.volAux.SetSize(nval_restr);
      operator_cache.volAux.UseDevice();
    }
    mfem::Vector &restr_grad_prim_dir(operator_cache.volAux);

    for(int idim = 0;idim < dim;idim++){

      mfem::Vector &grad_state_dir(*gradEntropy[idim]);
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

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::Mult(const mfem::Vector &u, mfem::Vector &dudt) const
  {
    ScopedTimer timer("NSRHS");

    const Vector &Ustate = u;


#ifdef SUBCELL_FV_BLENDING
    {
      ScopedTimer timer("SubcellBlendingComputeCoeff");
      // Since the CV are on-device, and computing
      // the indicator requires the CV, we compute
      // the indicator on-device and xfer only
      // alpha (the blending coeff) from host/device.
      ComputeIndicatorField(Ustate, operator_cache.indicatorField);
      ComputeBlendingCoefficientFromIndicator(operator_cache.indicatorField);
    }
#endif
    {
      ScopedTimer timer("Step");
      mfem::Vector &entropyState(operator_cache.entropyState);
      {
        ScopedTimer etime("EntropyPlumbing");
        if (entropyState.Size() != Ustate.Size()){
          entropyState.SetSize(Ustate.Size());
          entropyState.UseDevice();
        }
        ComputeEntropyState(Ustate, entropyState);
      }
      std::vector<mfem::Vector *> gradPrim(dim);
      {
        ScopedTimer gtime("GradientPlumbing");
        // grad_u is a vector of parallel grid functions
        // this bit grabs an mfem::Vector ref.
        // Note that incoming grad_u is really grad of entropy,
        // which we pack into gradPrim, and then call a
        // function which overwrites the entropy gradient
        // with the primitive gradient.
        for(int idim = 0;idim < dim;idim++){
          gradPrim[idim] = &(*grad_u[idim]);
        }
        GradOperator(entropyState, gradPrim);
        ComputeGradPrimFromGradEntropy(Ustate, gradPrim);
      }
      max_char_speed = MultCNS(Ustate, gradPrim, dudt);
    }
  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator_Volume(const mfem::Vector &pu,
                                                 std::vector<mfem::Vector *> &p_grad_u) const
  {
    ScopedTimer timer("GradOperator_Volume");

    const int dim = operator_cache.dim;
    const int restr_size = operator_cache.restr_v->Height();

    if(operator_cache.uVol.Size() != restr_size){
      operator_cache.uVol.SetSize(restr_size);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
    real_t *dU_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    real_t *pgrad_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    if (operator_cache.gradVol.size() != dim){
      operator_cache.gradVol.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        operator_cache.gradVol[idim].SetSize(restr_size);
        operator_cache.gradVol[idim].UseDevice();
      }
    }
    std::vector<mfem::Vector> &dUe(operator_cache.gradVol);
    for(int idim = 0;idim < dim;idim++){
      dU_d[idim] = dUe[idim].Write();
      pgrad_d[idim] = p_grad_u[idim]->Write();
    }

    mfem::forall(restr_size, [=] MFEM_HOST_DEVICE (int i)
    {
      for(int idim = 0;idim < dim;idim++){
        dU_d[idim][i] = real_t(0);
        pgrad_d[idim][i] = real_t(0);
      }
    });

    operator_cache.restr_v->Mult(pu, Ue);
    const real_t *Ue_d = Ue.Read();

    auto dc = device_cache;

    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int estride = ndof * neq;
    const int jac_stride = ndof;
    const int metric_stride = ndof * dc.dim * dc.dim;

    const real_t *elJac_d = dc.elJac_d;
    const real_t *elMetric_d = dc.elMetric_d;

    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      const real_t *u_el = Ue_d + e * estride;
      real_t *du_el_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        du_el_d[idim] = dU_d[idim] + e*estride;
      }

      const real_t *jac_el = elJac_d + e * jac_stride;
      const real_t *metric_el = elMetric_d + e * metric_stride;

      DGSEMIntegrator::AssembleGradElementVolumeKernel(dc, u_el, jac_el, metric_el,
                                                       du_el_d);
    });

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_v->AddMultTranspose(dUe[idim], *p_grad_u[idim]);
    }

  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator_BoundaryFaces(const mfem::Vector &pu,
                                                        std::vector<mfem::Vector *> &p_grad_u) const
  {
    ScopedTimer timer("GradOperator_BoundaryFaces_Device");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int face_size = nfp * neq;
    const int restr_size = operator_cache.restr_b->Height();
    const int nfaces_restr = restr_size / face_size;
    const int norm_size = nfp * dim;
    const int npoints_bnd = nfaces_restr * nfp;
    const int psize = pu.Size();

    mfem::Vector &u_faces(operator_cache.uBnd);
    if(u_faces.Size() != restr_size){
      u_faces.SetSize(restr_size);
      u_faces.UseDevice();
    }

    std::vector<mfem::Vector> &rhs_faces(operator_cache.gradBnd);
    if(rhs_faces.size() != dim){
      rhs_faces.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        rhs_faces[idim].SetSize(restr_size);
        rhs_faces[idim].UseDevice();
      }
    }

    mfem::Vector &duBnd(operator_cache.duBnd);
    if(duBnd.Size() != psize){
      duBnd.SetSize(psize);
      duBnd.UseDevice();
    }
    
    operator_cache.restr_b->Mult(pu, u_faces);
    
    const real_t *u_d = u_faces.Read();
    const real_t *nor_d = dc.bnd_nor_d;
    const real_t *wt_d = dc.bnd_wt_d;
    const int *bnd_marker_index_d = dc.bnd_marker_index_d;

    real_t *rhs_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    real_t *du_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      rhs_d[idim] = rhs_faces[idim].Write();
      du_d[idim] = duBnd.Write();
    }

    for (int idim = 0; idim < dim; ++idim) {
      real_t *rd = rhs_d[idim];
      mfem::forall(restr_size, [=] MFEM_HOST_DEVICE (int i) { rd[i] = real_t(0); });
    }
    for (int idim = 0; idim < 1; ++idim) {
      real_t *dud = du_d[idim];
      mfem::forall(psize, [=] MFEM_HOST_DEVICE (int i) { dud[i] = real_t(0); });
    }

    mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
    {
      const int f = p / nfp;
      const int fp = p % nfp;

      const int bnd_face_marker_index = bnd_marker_index_d[f];
      if (bnd_face_marker_index < 0)
        {
          return;
        }

      const int bc_index = bnd_face_marker_index; // same convention as inviscid device path for now
      if (bc_index < 0)
        {
          return;
        }

      const Prandtl::BCDescriptor &bc = dc.bc_descr_d[bc_index];
      if (bc.type == int(Prandtl::BCType::Invalid))
        {
          return;
        }

      const int face_offset = f * face_size;
      const int norm_offset = f * norm_size;
      const int w_offset = f * nfp;

      const real_t *u_face_d = u_d + face_offset;

      const real_t *nor_face_d = nor_d + norm_offset;
      const real_t *nor_point = nor_face_d + fp * dim;

      // Legacy one-sided boundary lifting uses +1/(w0*J1)
      const real_t scale = wt_d[w_offset + fp];

      real_t *rhs_face[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        rhs_face[idim] = rhs_d[idim] + face_offset;
      }

      DGSEMIntegrator::AssembleGradBoundaryPointKernel(dc, bc,
                                                       u_face_d,
                                                       nor_point,
                                                       scale,
                                                       fp,
                                                       rhs_face);
    });

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_b->MultTranspose(rhs_faces[idim], duBnd);
      *p_grad_u[idim] += duBnd;
    }

  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator_InteriorFaces(const mfem::Vector &pu,
                                                        std::vector<mfem::Vector *> &p_grad_u) const
  {
    ScopedTimer timer("GradOperator_InteriorFaces_Device");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int psize = pu.Size();
    const int restr_size = operator_cache.restr_f->Height();
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int nfaces = restr_size / (2 * nfp * neq);
    const int face_size = 2 * nfp * neq;
    const int norm_size = nfp * dim;

    mfem::Vector &u_faces(operator_cache.uInt);
    if(u_faces.Size() != restr_size){
      u_faces.SetSize(restr_size);
      u_faces.UseDevice();
    }

    std::vector<mfem::Vector> &rhs_faces(operator_cache.gradInt);
    if(rhs_faces.size() != dim){
      rhs_faces.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        rhs_faces[idim].SetSize(restr_size);
        rhs_faces[idim].UseDevice();
      }
    }

    mfem::Vector &duInt(operator_cache.duInt);
    if(duInt.Size() != psize){
      duInt.SetSize(psize);
      duInt.UseDevice();
    }

    operator_cache.restr_f->Mult(pu, u_faces);
    const real_t *u_d = u_faces.Read();

    real_t *rhs_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    real_t *du_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      rhs_d[idim] = rhs_faces[idim].Write();
      // We only need 1dim at a time, lets only use 1 temp
      du_d[idim] = duInt.Write();
    }

    for (int idim = 0; idim < dim; ++idim) {
      real_t *rd = rhs_d[idim];
      mfem::forall(restr_size, [=] MFEM_HOST_DEVICE (int i) { rd[i] = real_t(0); });
    }
    // HardCode to dim=1 for now - temporary is only 1d
    for (int idim = 0; idim < 1; ++idim) {
      real_t *dud = du_d[idim];
      mfem::forall(psize, [=] MFEM_HOST_DEVICE (int i) { dud[i] = real_t(0); });
    }

    const real_t *nor_d  = dc.nor_d;
    const real_t *wm_d   = dc.fw_minus_d;
    const real_t *wp_d   = dc.fw_plus_d;

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int f)
    {
      const int face_offset = f * face_size;
      const int norm_offset = f * norm_size;
      const int w_offset    = f * nfp;

      const real_t *u_face_d    = u_d + face_offset;
      const real_t *nor_face_d  = nor_d + norm_offset;
      const real_t *w_minus_d   = wm_d + w_offset;
      const real_t *w_plus_d    = wp_d + w_offset;

      real_t *rhs_face[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        rhs_face[idim] = rhs_d[idim] + face_offset;
      }

      DGSEMIntegrator::AssembleGradInteriorFaceKernel(dc,
                                                      u_face_d,
                                                      nor_face_d,
                                                      w_minus_d,
                                                      w_plus_d,
                                                      rhs_face);
    });

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_f->MultTranspose(rhs_faces[idim], duInt);
      *p_grad_u[idim] += duInt;
    }

  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator(const mfem::Vector &u,
                                          std::vector<mfem::Vector *> &grad_u) const
  {
    ScopedTimer timer("GradOperator_Device");
    const int dim = operator_cache.dim;
    const Vector &pu = Prolongate(u);
    std::vector<mfem::Vector *> p_grad_(dim);
    if (P)
      {
        const int psize = P->Height();
        if(operator_cache.pGrad.size() != dim){
          operator_cache.pGrad.resize(dim);
          for(int idim = 0;idim < dim;idim++){
            operator_cache.pGrad[idim].SetSize(psize);
            operator_cache.pGrad[idim].UseDevice();
          }
        }
        for(int idim = 0;idim < dim;idim++){
          p_grad_[idim] = &(operator_cache.pGrad[idim]);
        }
      }
    std::vector<mfem::Vector *> &p_grad_u = P ? p_grad_ : grad_u;

    MFEM_ASSERT(p_grad_u.size() == dim, "Size mismatch for gradient storage");
    MFEM_ASSERT(grad_u.size() == dim, "Size mismatch for gradient storage");

    GradOperator_Volume(pu, p_grad_u);

    GradOperator_InteriorFaces(pu, p_grad_u);

    GradOperator_BoundaryFaces(pu, p_grad_u);

    if (Serial())
      {
        if (cP)
          {
            for(int idim = 0;idim < dim;idim++){
              cP->MultTranspose(*p_grad_u[idim], *grad_u[idim]);
            }
          }
      }
    else
      {
        if(P){
          for(int idim = 0;idim < dim;idim++){
            P->MultTranspose(*p_grad_u[idim], *grad_u[idim]);
          }
        }
      }

    const int N = ess_tdof_list.Size();
    const auto idx = ess_tdof_list.Read();
    for(int idim = 0;idim < dim;idim++){
      auto gradu_dim_d = grad_u[idim]->ReadWrite();
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { gradu_dim_d[idx[i]] = 0.0; });
    }
  }

  template<typename PhysicsT>
  real_t NSOperator<PhysicsT>::MultCNS_InteriorFaces(const mfem::Vector &pu,
                                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                                 mfem::Vector &pdudt) const
{
  ScopedTimer timer("MultCNS_InteriorFaces");
  
  auto dc = device_cache;
  const int dim = dc.dim;
  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int nfaces = operator_cache.restr_f->Height() / (nfp * neq * 2); // (+/-)
  const int face_stride = 2 * nfp * neq;
  const int side_stride = nfp * neq;
  const int face_size = 2*nfp*neq;
  const int norm_size = nfp*dim;

  const int restr_size = operator_cache.restr_f->Height();
  mfem::Vector &int_u(operator_cache.uInt);
  if(int_u.Size() != restr_size){
    int_u.SetSize(restr_size);
    int_u.UseDevice();
  }

  mfem::Vector &rhs_faces(operator_cache.rhsInt);
  if(rhs_faces.Size() != restr_size){
    rhs_faces.SetSize(restr_size);
    rhs_faces.UseDevice();
  }

  mfem::Vector &faces_dudt(operator_cache.dudtInt);
  if(faces_dudt.Size() != pdudt.Size()){
    faces_dudt.SetSize(pdudt.Size());
    faces_dudt.UseDevice();
  }
  // faces_dudt = pdudt;

  const real_t *grad_prim_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  std::vector<mfem::Vector> &int_grad_prim(operator_cache.gradInt);

  if(int_grad_prim.size() != dim){
    int_grad_prim.resize(dim);
    for(int idim = 0;idim < dim;idim++){
      int_grad_prim[idim].SetSize(restr_size);
      int_grad_prim[idim].UseDevice();
    }
  }

  for(int idim = 0;idim < dim;idim++){
    operator_cache.restr_f->Mult(*p_grad_prim[idim], int_grad_prim[idim]);
    grad_prim_d[idim] = int_grad_prim[idim].Read();
  }
  
  // If zeroed before accumulation, do it explicitly on device:
  // Potentially, this is not needed at all since I think we overwrite everything
  {
    real_t *d = rhs_faces.Write();
    mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  operator_cache.restr_f->Mult(pu, int_u);

  const real_t *u_d = int_u.Read();
  real_t *rhs_d = rhs_faces.Write();
  const real_t *nor_d   = dc.nor_d;      // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.fw_minus_d; // size nfaces*nfp
  const real_t *inv2_d  = dc.fw_plus_d;  // size nfaces*nfp

  real_t *ws_d = dc.ifWaveSpeed_d;

  mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int i)
  {
    const int face_offset = i*face_size;
    const int n_offset = i*norm_size;
    const int w_offset = i*nfp;

    const real_t *u_face_d = u_d + face_offset;
    real_t *rhs_face_d = rhs_d + face_offset;
    const real_t *nor_face_d = nor_d + n_offset;
    const real_t *w_minus_d = inv1_d + w_offset;
    const real_t *w_plus_d = inv2_d + w_offset;
    const real_t *dprim_face_x = (dim > 0) ? grad_prim_d[0] + face_offset : nullptr;
    const real_t *dprim_face_y = (dim > 1) ? grad_prim_d[1] + face_offset : nullptr;
    const real_t *dprim_face_z = (dim > 2) ? grad_prim_d[2] + face_offset : nullptr;

    // Call one fused kernel for inviscid and viscous facial terms
    real_t ws = DGSEMIntegrator::AssembleViscousElementFaceKernel(dc, u_face_d, nor_face_d,
                                                                  w_minus_d, w_plus_d,
                                                                  dprim_face_x,
                                                                  dprim_face_y,
                                                                  dprim_face_z,
                                                                  rhs_face_d);
    ws_d[i] = ws;
  });

  operator_cache.restr_f->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt; // on device? 

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = operator_cache.ifWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int f = 0;f < operator_cache.num_interior_faces;f++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[f]);
    }

  return max_char_speed_facial;
}


  template<typename PhysicsT>
  real_t NSOperator<PhysicsT>::MultCNS_BoundaryFaces(const mfem::Vector &pu,
                                                     const std::vector<mfem::Vector *> &p_grad_prim,
                                                     mfem::Vector &pdudt) const
{
  ScopedTimer timer("MultCNS_BoundaryFaces");

  auto dc = device_cache;
  const int dim = dc.dim;
  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int face_size = nfp * neq;
  const int restr_size = operator_cache.restr_b->Height();
  const int nfaces_restr = restr_size / face_size;
  const int norm_size = nfp * dc.dim;
  const int npoints_bnd = nfaces_restr * nfp;
  const int psize = pdudt.Size();

  mfem::Vector &rhs_faces(operator_cache.rhsBnd);
  mfem::Vector &faces_dudt(operator_cache.dudtBnd);
  if(rhs_faces.Size() != restr_size){
    rhs_faces.SetSize(restr_size);
    rhs_faces.UseDevice();
  }
  if(faces_dudt.Size() != psize){
    faces_dudt.SetSize(psize);
    faces_dudt.UseDevice();
  }
  mfem::Vector &bnd_u(operator_cache.uBnd);
  if(bnd_u.Size() != restr_size){
    bnd_u.SetSize(restr_size);
    bnd_u.UseDevice();
  }
  const real_t *grad_prim_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  std::vector<mfem::Vector> &bnd_grad_prim(operator_cache.gradBnd);
  if(bnd_grad_prim.size() != dim){
    bnd_grad_prim.resize(dim);
    for(int idim = 0;idim < dim;idim++){
      bnd_grad_prim[idim].SetSize(restr_size);
      bnd_grad_prim[idim].UseDevice();
    }
  }
  for(int idim = 0;idim < dim;idim++){
    operator_cache.restr_b->Mult(*p_grad_prim[idim], bnd_grad_prim[idim]);
    grad_prim_d[idim] = bnd_grad_prim[idim].Read();
  }

  // If zeroed before accumulation, do it explicitly on device:
  // Potentially, this is not needed at all since I think we overwrite everything
  {
    real_t *rd = rhs_faces.Write();
    mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i)
    { rd[i] = real_t(0);});
    real_t *fd = faces_dudt.Write();
    mfem::forall(faces_dudt.Size(), [=] MFEM_HOST_DEVICE (int i)
    { fd[i] = real_t(0);});
  }

  operator_cache.restr_b->Mult(pu, bnd_u);

  const real_t *u_d = bnd_u.Read();
  real_t *rhs_d = rhs_faces.Write();

  const real_t *nor_d   = dc.bnd_nor_d;      // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.bnd_wt_d; // size nfaces*nfp
  const int *bnd_marker_index_d = dc.bnd_marker_index_d;
  real_t *ws_d = dc.bndWaveSpeed_d;

  mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
  {
    const int f = p / nfp;
    const int fp = p % nfp;

    int bnd_face_marker_index = bnd_marker_index_d[f];
    if(bnd_face_marker_index < 0){
      ws_d[p] = 0.0;
      return;
    }
    int bc_index = bnd_face_marker_index; // no mapping atm
    if(bc_index < 0){
      ws_d[p] = 0.0;
      return;
    }
    const Prandtl::BCDescriptor &bc = dc.bc_descr_d[bc_index];
    if (bc.type == int(Prandtl::BCType::Invalid))
      {
        ws_d[p] = 0.0;
        return;
      }

    const int face_offset = f * face_size;
    const int n_offset = f * norm_size;
    const int w_offset = f * nfp;

    const real_t *u_face_d = u_d + face_offset;
    real_t *rhs_face_d = rhs_d + face_offset;
    const real_t *nor_face_d = nor_d + n_offset;
    const real_t *w_minus_d = inv1_d + w_offset;
    const real_t *nor_point = nor_face_d + fp*dim;
    real_t scale = -w_minus_d[fp];
    // #ifdef AXISYMMETRIC
    // NOTE: axisymmetric not ready for device yet
    // scale *= rad_face[fp];
    // #else
    // #error "Axisymmetric boundary device path not implemented yet."
    // #endif
    real_t state1[Prandtl::MAXEQ];
    real_t fluxN[Prandtl::MAXEQ];
    real_t gradPrim_x[Prandtl::MAXEQ];
    real_t gradPrim_y[Prandtl::MAXEQ];
    real_t gradPrim_z[Prandtl::MAXEQ];
    const real_t *dprim_face_x = (dim > 0) ? grad_prim_d[0] + face_offset : nullptr;
    const real_t *dprim_face_y = (dim > 1) ? grad_prim_d[1] + face_offset : nullptr;
    const real_t *dprim_face_z = (dim > 2) ? grad_prim_d[2] + face_offset : nullptr;
    Prandtl::Kernels::el_gather_grad_state(dprim_face_x, dprim_face_y, dprim_face_z,
                                           dim, nfp, neq, fp, gradPrim_x, gradPrim_y,
                                           gradPrim_z);
    Prandtl::Kernels::el_gather_state(u_face_d, nfp, neq, fp, state1);
    
    const real_t ws = \
      Prandtl::BC::ApplyViscousBoundaryCondition(dc, bc, state1, gradPrim_x, gradPrim_y,
                                                 gradPrim_z, nor_point, fluxN);
    Prandtl::Kernels::el_scatter_add(fluxN, nfp, neq, fp, scale, rhs_face_d);
    ws_d[p] = ws;
  });

  operator_cache.restr_b->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt; // on device? (likely yes) 

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = operator_cache.bndWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int p = 0;p < npoints_bnd;p++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[p]);
    }

  return max_char_speed_facial;
}

  template<typename PhysicsT>
  real_t NSOperator<PhysicsT>::MultCNS_Volume(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                                              mfem::Vector &pdudt) const
{
  ScopedTimer timer("MultCNS_Volume");
  // Copy the device cache so that it is not member data
  auto dc = device_cache;
  const int dim = dc.dim;
  const int restr_size = operator_cache.restr_v->Height();

  mfem::Vector &vol_u(operator_cache.uVol);
  if(vol_u.Size() != restr_size){
    vol_u.SetSize(restr_size);
    vol_u.UseDevice();
  }

  mfem::Vector &dUe(operator_cache.rhsVol);
  if(dUe.Size() != restr_size){
    dUe.SetSize(restr_size);
    dUe.UseDevice();
  }

  std::vector<mfem::Vector> &vol_grad_prim(operator_cache.gradVol);
  if(vol_grad_prim.size() != dim){
    vol_grad_prim.resize(dim);
    for(int idim = 0;idim < dim;idim++){
      vol_grad_prim[idim].SetSize(restr_size);
      vol_grad_prim[idim].UseDevice();
    }
  }

  operator_cache.restr_v->Mult(pu, vol_u);
  for(int idim = 0;idim < dim;idim++){
    operator_cache.restr_v->Mult(*p_grad_prim[idim], vol_grad_prim[idim]);
  }

  // Zero the RHS array on-device
  {
    real_t *d = dUe.Write();
    mfem::forall(dUe.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  // Set up the read-only pointers for restr inputs
  const real_t *Ue_d = vol_u.Read();
  const real_t *gradPrim_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  for(int idim = 0;idim < dim;idim++){
    gradPrim_d[idim] = vol_grad_prim[idim].Read();
  }
  
  // Write-only for RHS
  real_t *dUe_d = dUe.Write();

  // Device cache parameters
  const int ne = dc.num_elements;
  const int ndof = dc.ndof_scalar_el;
  const int neq = dc.num_equations;

#ifdef SUBCELL_FV_BLENDING
  const int Np_x = dc.Np_x;
  const int Np_y = dc.Np_y;
  const int Np_z = dc.Np_z;
  const int npe = Np_x * Np_y * Np_z;
  const int ndofe = npe * neq;
  const int npe_metric_xi = (Np_x + 1)*Np_y*Np_z;
  const int npe_metric_eta = Np_x*(Np_y + 1)*Np_z;
  const int npe_metric_zeta = Np_x * Np_y * (Np_z + 1);
  const real_t *metric_xi_d = dc.subcell_metric_xi_d;
  const real_t *metric_eta_d = (dim > 1 ? dc.subcell_metric_eta_d : nullptr);
  const real_t *metric_zeta_d = (dim > 2 ? dc.subcell_metric_zeta_d : nullptr);

  mfem::Vector dUfv(operator_cache.restr_v->Height());
  dUfv.UseDevice();
  real_t *dUfv_d = dUfv.Write();
  // zero the array on-device
  {
    real_t *d = dUfv_d;
    mfem::forall(dUfv.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  const real_t *alpha_d = operator_cache.alpha->Read();
#endif

  // Derived parameters
  const int metric_stride = ndof * dim * dim;
  const int jac_stride    = ndof;
  const int estride = ndof*neq;
  
  // Device cache data/arrays
  const int *elem_attr_d = dc.elem_attr_d;
  const int *attr_marker_d = dc.attr_marker_d;
  const real_t *elJac_d = dc.elJac_d;
  const real_t *elMetric_d = dc.elMetric_d;

  real_t *ws_d = dc.elWaveSpeed_d;

  // Inside the FORALL below, executed on device
  mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
  {
    
    const real_t *jac_el    = elJac_d    + e * jac_stride;
    const real_t *metric_el = elMetric_d + e * metric_stride;

    const int attr = elem_attr_d[e];
    if (attr_marker_d[attr-1] == 0) {
      ws_d[e] = 0.0;
      return;
    }

    // Element-specific inputs and outputs
    const int eoff = e * estride;
    const real_t *u_el = Ue_d + eoff;
    real_t *du_el = dUe_d + eoff;

    real_t cs_el = \
      DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                   jac_el, metric_el, du_el);
#ifdef SUBCELL_FV_BLENDING
    real_t alpha_fv = alpha_d[e];
    if(alpha_fv > 1e-16){
      real_t alpha_dg = (1.0 - alpha_fv);
      real_t *du_fv = dUfv_d + eoff;
      const real_t *el_metric_xi = metric_xi_d + e * npe_metric_xi * dim;
      const real_t *el_metric_eta = (dim > 1 ? metric_eta_d + e * npe_metric_eta * dim :
                                     nullptr);
      const real_t *el_metric_zeta = (dim > 2 ? metric_zeta_d + e * npe_metric_zeta * dim :
                                      nullptr);
      const real_t cs_fv =                                              \
        DGSEMIntegrator::ComputeFVFluxesKernel(dc, u_el, jac_el, el_metric_xi, el_metric_eta, el_metric_zeta, du_fv);
      
      for(int ipt = 0;ipt < estride;ipt++){
        du_el[ipt] = alpha_dg * du_el[ipt] + alpha_fv * du_fv[ipt];
      }

      cs_el = Kernels::rmax(cs_el, cs_fv);
    }
#endif
    ws_d[e] = cs_el;

    // Inviscid part is done: dUe currrently holds the inviscid RHS
    // Host code mixes inviscid and viscous assembly, we need separate.
    // Call the Viscous Assembly routine
    const real_t *grad_prim_el[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      grad_prim_el[idim] = gradPrim_d[idim] + eoff;
    }

    DGSEMIntegrator::AssembleViscousElementVolumeKernel(dc, u_el, jac_el, metric_el,
                                                        grad_prim_el[0], grad_prim_el[1],
                                                        grad_prim_el[2], du_el);

  });

  // The rest is identical to Euler operator
  // Scatter RHS back to storage
  operator_cache.restr_v->AddMultTranspose(dUe, pdudt);

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = operator_cache.elWaveSpeed.HostRead();
  real_t max_char_speed = 0.0;
  for(int e = 0;e < operator_cache.num_elements;e++)
    {
      max_char_speed = std::max(max_char_speed, ws[e]);
    }

  return max_char_speed;
}

template<typename PhysicsT>
real_t NSOperator<PhysicsT>::MultCNS(const mfem::Vector &u, const std::vector<mfem::Vector *> &grad_prim,
                                     mfem::Vector &dudt) const
{
  const int dim = operator_cache.dim;
  const Vector &pu = Prolongate(u);
  std::vector<mfem::Vector *> p_grad_(dim);
  if (P)
    {
      const int psize = P->Height();
      if(operator_cache.pGrad.size() != dim){
        operator_cache.pGrad.resize(dim);
        for(int idim = 0;idim < dim;idim++){
          operator_cache.pGrad[idim].SetSize(psize);
          operator_cache.pGrad[idim].UseDevice();
        }
      }
      for(int idim = 0;idim < dim;idim++){
        p_grad_[idim] = &(operator_cache.pGrad[idim]);
        P->Mult(*grad_prim[idim], *p_grad_[idim]);
      }
      if(operator_cache.pdudt.Size() != psize){
        operator_cache.pdudt.SetSize(psize);
        operator_cache.pdudt.UseDevice();
      }
    }
  const std::vector<mfem::Vector *> &pGradPrim = P ? p_grad_ : grad_prim;
  Vector &pdudt = P ? operator_cache.pdudt : dudt;
  pdudt = 0.0;
  
  real_t max_char_speed = MultCNS_Volume(pu, pGradPrim, pdudt);

  real_t max_char_speed_faces = MultCNS_InteriorFaces(pu, pGradPrim, pdudt);
  max_char_speed = std::max(max_char_speed, max_char_speed_faces);

  real_t max_char_speed_bnd = MultCNS_BoundaryFaces(pu, pGradPrim, pdudt);
  max_char_speed = std::max(max_char_speed, max_char_speed_bnd);
  
  if (Serial())
    {
      if (cP)
        {
          cP->MultTranspose(pdudt, dudt);
        }

    }
  else
    {
      P->MultTranspose(pdudt, dudt);
    }

  const int N = ess_tdof_list.Size();
  const auto idx = ess_tdof_list.Read();
  auto DU_RW = dudt.ReadWrite();
  mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

  return max_char_speed;
}

}
