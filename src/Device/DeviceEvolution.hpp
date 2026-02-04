#pragma once

namespace Prandtl::Device {
  static constexpr int MAX_EQ = 5;
  struct PhysicsConstants
  {
    real_t gamma;
    real_t gammaInverse;
    real_t gammaP1;
    real_t gammaM1;
    real_t gammaP1Inverse;
    real_t gammaM1Inverse;
    real_t gamma_gammaM1Inverse; // gamma * gammaM1Inverse;
    real_t gammaM1_gammaInverse; // gammaM1 * gammaInverse;
    real_t Pr;
    real_t PrInverse;
    real_t R_gas;
    real_t cp;
    real_t mu;
    real_t mu_bulk = 2.0 / 3.0;
    real_t mu0 = 1.716e-5;
    real_t T0 = 273.15;
    real_t Ts = 110.4;
  };
  
  struct StateLayout
  {
    int dim = -1;              // spatial dimension (1,2,3)
    int num_dofs_scalar = -1;  // DOFs per scalar field (block length)
    
    // Equation indices (0-based)
    int eq_mass = -1;          // mass density
    int eq_mom[3] = {-1, -1, -1};        // momentum density components: x,y,z
    int eq_energy = -1;        // total energy density

    // Optional scalar support (forward-looking)
    int eq_scalar0 = -1;       // index of first scalar component (or -1 if none)
    int num_scalars = -1;      // number of scalar components
    
    /**
     * Set up after creation.
     *
     * This method exists to allow default construction of StateLayout objects,
     * which is sometimes required for use in containers (e.g., std::vector),
     * serialization frameworks, or APIs that require default-constructible types.
     * In such cases, the object is first default-constructed and then initialized
     * via this setup() method.
     *
     * Prefer using the parameterized constructor whenever possible to ensure
     * objects are fully initialized at construction time. Use setup() only when
     * default construction is unavoidable due to external constraints.
     *
     * @param dim_             Spatial dimension (1, 2, or 3)
     * @param num_dofs_scalar_ Number of DOFs per scalar field
     * @param num_scalars_     Number of scalar components (default: 0)
     */
    MFEM_HOST_DEVICE void setup(int dim_, int num_dofs_scalar_, int num_scalars_ = 0)
    {
      dim = dim_;
      num_dofs_scalar = num_dofs_scalar_;
      eq_mass = 0;
      eq_mom0 = 1;
      eq_energy = dim_ + 1;
      eq_scalar0 = (num_scalars_ > 0 ? (dim_ + 2) : -1);
      num_scalars = num_scalars_;
      // Momentum components follow mass
      for (int d = 0; d < 3; ++d)
        {
          eq_mom[d] = (d < dim_) ? (1 + d) : -1;
        }
    }

    // Convenience for nequations
    MFEM_HOST_DEVICE inline int nequations() const
    { return dim + 2 + num_scalars; }

    // parameter validation routine
    MFEM_HOST_DEVICE inline int validate(int equation, int dof) const
    {
      return ((equation < nequations() && dof < num_dofs_scalar &&
               equation > -1 && dof > -1) ? 0 : 1);
    }

    // Flat index into the equation-blocked vector
    MFEM_HOST_DEVICE inline int index(int equation, int dof) const
    {
      // assert(validate(equation, dof) == 0);
      return equation * num_dofs_scalar + dof;
    }

    MFEM_HOST_DEVICE inline 
  };

// ============================================================================
// EOS: Ideal single-species gas using PhysicsConstants
// ============================================================================
  struct IdealSingleGasEOS
  {
    // ---- helpers on conservative state --------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t R_gas(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return phys.R_gas;
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t density(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
        return S.mass(L); // this is "rho" (mass density)
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t rhoE(const PhysicsConstants &phys, const StateLayout &L,
                       const StateView &S) const
    {
        return S.energy(L);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t momentum_sq(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
        const int dim = L.dim;   // uses state layout
        real_t m2 = 0;
        for (int d = 0; d < dim; ++d)
        {
          const real_t m = S.momentum(L,d);
          m2 += m * m;
        }
        return m2;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                         const StateView &S) const
    {
      // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
      const real_t rho  = density(phys, L, S);
      const real_t m2   = momentum_sq(phys, L, S);
      return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                          const StateView &S) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return rhoE(phys, L, S) - kinetic_energy_density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_from_pressure(const PhysicsConstants &phys, const StateLayout &L,
                                                const StateView &S, real_t pressure) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
      return pressure / (phys.gamma - 1.0);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                           const StateView &S) const
    {
        // e = (rho*e) / rho
      const real_t rho  = density(phys, L, S);
      const real_t rhoe = internal_energy_density(phys, L, S);
      return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t pressure(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S) const
    {
      // p = (gamma - 1) * (rho*E - 0.5*|rho*u|^2 / rho)
      const real_t rhoe = internal_energy_density(phys, L, S);
      return phys.gammaM1 * rhoe;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t gamma(const PhysicsConstants &phys, const StateLayout &L,
                        const StateView &S) const
    {
      return phys.gamma;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t temperature(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      // p = rho*R*T  =>  T = p / (rho*R)
      const real_t rho = density(phys, L, S);
      const real_t p   = pressure(phys, L, S);
      return p / (rho * phys.R_gas);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const PhysicsConstants &phys, const StateLayout  &L,
                                 const StateView &S, const real_t *grad_rho,
                                 const real_t *grad_p, real_t *grad_t) const
    {
      const int dim = L.dim;
      const real_t rho = density(phys, L, S);
      const real_t pressor = pressure(phys, L, S)/rho;
      const real_t cv = cp(phys, L, S)/phys.gamma;
      const real_t fac = phys.gammaM1Inverse/(cv*rho);
      for(int i = 0; i < dim; i++){
        grad_t[i] = fac*(grad_p[i] - pressor*grad_rho[i]);
      }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S) const
    {
      // a^2 = gamma * p / rho
      const real_t rho = density(phys, L, S);
      const real_t p   = pressure(phys, L, S);
      return std::sqrt(phys.gamma * p / rho);
    }
    
    // cp is constant for ideal gas
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t cp(const PhysicsConstants &phys, const StateLayout &L,
                     const StateView & /*S*/) const
    {
        return phys.cp;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t entropy(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S) const
    {
      const real_t p = pressure(phys, L, S);
      const real_t gamma = phys.gamma;
      // TODO: Augment for correct treatment of passive scalars
      return std::log(p) - gamma * std::log(S.mass(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const PhysicsConstants &phys, const StateLayout &L,
                              const InStateView &S, OutStateView &E) const
    {
      const real_t p = pressure(phys, L, S);
      const real_t gamma = phys.gamma;
      const real_t rho = S.mass(L);
      const real_t s = std::log(p) - gamma*std::log(rho);
      const real_t beta = rho / p;
      const real_t v2o2 = kinetic_energy_density(phys, L, S) / rho;
      const real_t s_rho = (gamma - s)/(gamma - 1) - beta*v2o2;

      E.set_mass(L, s_rho);
      int dim = L.dim;
      int num_scalars = L.num_scalars;
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(L, idim, beta * S.velocity(L, idim));
      }
      E.set_energy(L, -beta);
      // TODO: Update for correct treatment of passive scalars (depends on ES approach)
      // - Here we should probably set the entropy state to scalar_state / density
      // - If we do that, we need to modify the mass component of the entropy state
      // - Making this fix will make the sensor function sensitive to the scalars
      // - If we need to recover CV from this, lax scalar treatment is a nogo
      for(int iscalar = 0;iscalar < num_scalars;iscalar++){
        E.set_scalar(L, iscalar, 0.0);
      }
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const PhysicsConstants &phys, const StateLayout &L,
                                          const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim) const
    {

      const real_t ke = kinetic_energy_density(phys, L, S);
      const real_t p = pressure(phys, L, S);
      const real_t rho = S.mass(L);
      const real_t rhoE = S.energy(L);
      const real_t ie = internal_energy_density(phys, L, S);

      int dim = L.dim;
      int num_scalars = L.num_scalars;

      real_t drho = 0.0;
      for(int idim = 0; idim < dim; idim++){
        dPrim.set_momentum(L, idim, p/rho * (dE.momentum(L, idim) + S.velocity(L, idim)*dE.energy(L)));
        drho += S.momentum(L, idim)*dPrim.momentum(L, idim);
      }
      drho = rho*dE.mass(L) - dE.energy(L)*(ke - ie) + rho*drho/p;
      dPrim.set_mass(L, drho);
      dPrim.set_energy(L, p/rho * (dPrim.mass(L) + p*dE.energy(L)));
      for(int isp = 0; isp < num_scalars; isp++){
        dPrim.set_scalar(L, isp, 0.0); // just a placeholder for now
      }
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                     const InStateView &Se, OutStateView &Sc) const
    {
      int dim = L.dim;
      const real_t beta = -Se.energy(L);
      real_t k = 0.0;
      real_t vel[3];
      for(int idim = 0;idim < dim;idim++){
        vel[idim] = Se.momentum(L, idim)/beta;
        k += vel[idim]*vel[idim];
      }
      const real_t gamma = phys.gamma;
      const real_t s = gamma - (Se.mass(L) + 0.5*k*beta)*(gamma - 1.);
      const real_t rho = std::pow(std::exp(-s)/beta, 1.0/(gamma - 1));
      Sc.set_mass(L, rho);
      Sc.set_energy(L, rho*(1.0/(beta*(gamma-1.)) + 0.5*k));
      for(int idim = 0;idim < dim;idim++){
        Sc.set_momentum(L, idim, rho*vel[idim]);
      }
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const PhysicsConstants &phys, const StateLayout &L,
                         const StateView &S, real_t u[3]) const
    {
      const int dim = L.dim;
      for (int d = 0; d < dim; ++d)
        {
          u[d] = S.velocity(L, d);
        }
      for (int d = dim; d < 3; ++d)
        {
          u[d] = real_t(0);
        }
    }
  };

// ============================================================================
// GasModel: Encapsulate EOS/Transport
// ============================================================================
  template <typename EOSImpl, typename TransportImpl>
  struct GasModel
  {

    PhysicsConstants phys;
    StateLayout L;
    EOSImpl eos;
    TransportImpl transport;
    
    MFEM_HOST_DEVICE
    GasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
             const EOSImpl &eos_in, const TransportImpl &tr_in)
      : phys(phys_in), L(L_in), eos(eos_in), transport(tr_in)
    { }

    MFEM_HOST_DEVICE
    GasModel(const PhysicsConstants &phys_in, const StateLayout &L_in)
      : phys(phys_in), L(L_in)
    { }

    // Utilities and constants etc
    MFEM_HOST_DEVICE
    inline int num_equations() const
    { return L.nequations(); };

    MFEM_HOST_DEVICE
    inline int dim() const
    { return L.dim; };

    // State Access
    MFEM_HOST_DEVICE
    template<typename StateView>
    inline real_t velocity(const StateView &S, int d) const
    { return S.velocity(L,d);}

    MFEM_HOST_DEVICE
    template<typename StateView>
    inline real_t momentum(const StateView &S, int d) const
    { return S.momentum(L,d);}

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t density(const StateView &S) const
    {
      return eos.density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t mass(const StateView &S) const
    {
      return eos.density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t energy(const StateView &S) const
    {
      return S.energy(L);
    }

    // --- Thermodynamics ------------------------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t pressure(const StateView &S) const
    {
      return eos.pressure(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t gamma(const StateView &S) const
    {
      return eos.gamma(phys, L, S);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t cp(const StateView &S) const
    {
      return eos.cp(phys, L, S);
    }
    
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t R_gas(const StateView &S) const
    {
      return eos.R_gas(phys, L, S);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t temperature(const StateView &S) const
    {
      return eos.temperature(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t sound_speed(const StateView &S) const
    {
      return eos.sound_speed(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t kinetic_energy_density(const StateView &S) const
    {
      return eos.kinetic_energy_density(phys, L, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t internal_energy_from_pressure(const StateView &S, real_t pressure) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
      return eos.internal_energy_from_pressure(phys, L, S, pressure);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t specific_internal_energy(const StateView &S) const
    {
      return eos.specific_internal_energy(phys, L, S);
    }
    
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const StateView &S,
                                 const real_t *grad_r, const real_t *grad_p,
                                 real_t *grad_t) const
    {
      return eos.grad_temperature(phys, L, S, grad_r, grad_p, grad_t);
    }
 
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t entropy(const StateView &S)
    {
      return eos.entropy(phys, L, S);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const InStateView &S, OutStateView &E) const
    {
      return eos.entropy_state(phys, L, S, E);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const InStateView &S, const InStateView &dS,
                                          OutStateView &dPrim) const
    {
      return eos.grad_entropy_to_grad_prim(phys, L, S, dS, dPrim);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const InStateView &Se, OutStateView &Sc) const
    {
      return eos.entropy_to_conserved(phys, L, Se, Sc);
    }

    // --- Transport -----------------------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t viscosity(const StateView &S) const
    {
      return transport.viscosity(phys, L, eos, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t bulk_viscosity(const StateView &S) const
    {
      return transport.bulk_viscosity(phys, L, eos, S);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline real_t thermal_conductivity(const StateView &S) const
    {
      return transport.thermal_conductivity(phys, L, eos, S);
    }
  };

  // Current concrete choice: ideal single-species gas + constant transport
  using IdealGasModel = GasModel<IdealSingleGasEOS, Transport>;
  MFEM_HOST_DEVICE inline real_t rmax(real_t a, real_t b) { return a > b ? a : b; }
  MFEM_HOST_DEVICE inline real_t rsqrt(real_t x) { return ::sqrt(x); }  // mfem::sqrt?
  MFEM_HOST_DEVICE inline real_t rlog(real_t x)  { return ::log(x); }   // mfe::log?

  MFEM_HOST_DEVICE
  inline void ComputeMeanVec(const real_t* a, const real_t* b, real_t* out, int n)
  {
    for (int i=0;i<n;++i) out[i] = real_t(0.5)*(a[i]+b[i]);
  }

  MFEM_HOST_DEVICE
  inline real_t ComputeLogMean(real_t x, real_t y, real_t eps) // eps defaults to 1e-4 on CPU
  {
    const real_t xi = y / x;
    const real_t u  = (xi*(xi - 2.0) + 1.0) / (xi*(xi + 2.0) + 1.0);
    
    // polynomial approximation branch when u is small
    if (u < eps)
      {
        // (x+y)*52.5 / (105 + u*(35 + u*(21 + 15*u)))
        const real_t denom = 105.0 + u*(35.0 + u*(21.0 + 15.0*u));
        return (x + y) * 52.5 / denom;
      }
    else
      {
        return (y - x) / Prandtl::Device::rlog(xi);
      }
  }

  struct DGSEMCache
  {
    // Host-only handles used for gather/scatter
    mfem::ElementRestriction *restr_v = nullptr;   // for vfes (vector space)
    mfem::ElementRestriction *restr_s = nullptr;   // for fes (scalar space)
    int num_el = 0;
    int ndof_scalar_el = 0;
    int num_eq = 0;
    int num_attr = 0;
    // Persistent device-resident aux arrays
    mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
    mfem::Array<int> attr_marker;  // size nattr, 0/1
    mfem::Vector elJac;
    mfem::Vector elMetric;
    mfem::Vector elWaveSpeed;
    // const real_t *D;
  };
  
  MFEM_HOST_DEVICE
  inline real_t ComputeVolumeFlux_Chandrashekar(const auto& gasModel, const real_t* q1,
                                                const real_t* q2,
                                                const real_t* met1,
                                                const real_t* met2,
                                                real_t* F_tilde)
  {
    const int dim = gasModel.dim();
    const int neq = gasModel.num_equations();

    // mean metric row
    real_t met[3] = {0,0,0};
    ComputeMeanVec(met1, met2, met, dim);

    const real_t rho1 = gasModel.density(q1);
    const real_t rho2 = gasModel.density(q2);
    const real_t rho_ln = ComputeLogMean(rho1, rho2, 1e-4);

    real_t mom_hat[3] = {0,0,0};
    real_t h_hat = 0;
    real_t vn = 0;
    real_t v2_1 = 0;
    real_t v2_2 = 0;

    for (int d=0; d<dim; ++d)
      {
        const real_t v1 = gasModel.velocity(q1, d);
        const real_t v2 = gasModel.velocity(q2, d);
        const real_t vbar = real_t(0.5)*(v1+v2);
        
        v2_1 += v1*v1;
        v2_2 += v2*v2;
        vn   += vbar * met[d];
        
        mom_hat[d] = rho_ln * vbar;

        h_hat += -real_t(0.25)*(v1*v1 + v2*v2) + vbar*vbar;
      }
    
    
    const real_t p1 = gasModel.pressure(q1);
    const real_t p2 = gasModel.pressure(q2);

    const real_t speed1 = Prandtl::Device::rsqrt(v2_1);
    const real_t speed2 = Prandtl::Device::rsqrt(v2_2);

    const real_t c1 = gasModel.sound_speed(q1);
    const real_t c2 = gasModel.sound_speed(q2);

    const real_t lambda_max = Prandtl::Device::rmax(speed1 + c1, speed2 + c2);

    // Single-component ideal-gas-specific KEPEC bits
    // TODO: Update/Craft KPEC fluxes for mixtures (and passive scalar components)
    const real_t beta1 = real_t(0.5) * rho1 / p1;
    const real_t beta2 = real_t(0.5) * rho2 / p2;
    const real_t beta_ln = ComputeLogMean(beta1, beta2, ctx.logmean_eps);
    
    const real_t p_hat = real_t(0.5) * (rho1 + rho2) / (beta1 + beta2);

    const real_t gm11 = ctx.gas.gamma(q1);
    const real_t gm12 = ctx.gas.gamma(q2);
    const real_t gm1_av_inv = real_t(2.0) / (gm11 + gm12 - real_t(2.0));

    h_hat += real_t(0.5) / beta_ln * gm1_av_inv + p_hat / rho_ln;

    // F_tilde layout: [rho, rhoV, rhoE]
    // NOTE: Caller *must* zero(or own) F_tilde (size: neq)
    F_tilde[0] = rho_ln * vn;
    for (int d=0; d<dim; ++d)
      {
        F_tilde[1 + d] = vn * mom_hat[d] + p_hat * met[d];
      }
    F_tilde[1 + dim] = rho_ln * vn * h_hat;

  // TODO: Updte for scalars, sigh
  // for (s=0; s<num_scalars; ++s) F_tilde[XXXX]= XXX

  return lambda_max;
}

  
  void DGSEMIntegrator::AssembleGeometricTerms()
  {
    if(use_partial_assembly){
      int nelem = fes0->GetNE();
      assert(nelem == num_elements);
      elJac.SetSize(Np_x*Np_y*Np_z*num_elements);
      elMetric.SetSize(dim*dim*Np_x*Np_y*Np_z*num_elements);

      std::cout << "AssembleGeometricTerms::Number of elements: " << nelem << std::endl;

      for (int i = 0; i < nelem; i++)
        {
          ElementTransformation *T = fes0->GetElementTransformation(i);
          assert(T->ElementNo == i);
          AssembleElementGeometricTerms(*T);
        }
    }
  }

void DGSEMIntegrator::AssembleElementGeometricTerms(ElementTransformation &Tr)
{

  if (debug_integrator)
    {
      std::cout << "===== Entering DGSEMIntegrator::AssembleElementGeometricalTerms =====" << std::endl;
    }
  
  real_t *Jinv_h = elJac.HostWrite();
  real_t *Met_h  = elMetric.HostWrite();

  const int e = Tr.ElementNo;
  const int nq = Np_x * Np_y * Np_z;
  
  for (int q = 0; q < nq; ++q)
    {
      const IntegrationPoint &ip = ir_vol->IntPoint(q);
      Tr.SetIntPoint(&ip);
      const real_t J = Tr.Weight();
      Jinv_h[e*nq + q] = J;
      
      const mfem::DenseMatrix &adj = Tr.AdjugateJacobian();              
      for (int dir = 0; dir < dim; ++dir)
        {
          // adj.GetRow(dir, metricRow) pattern:
          // metricRow is a mfem::Vector of size dim you already have (e.g. member 'metric1')
          adj.GetRow(dir, metric1);  // metric1.Size() == dim
          
          for (int d = 0; d < dim; ++d)
            {
              const int idxM = (((e*nq + q)*dim + dir)*dim + d);
              Met_h[idxM] = metric1(d);
            }
        }
    }
  elJac.Read();
  elMetric.Read();
}

  // Originally DGSEMIntegrator.cpp::AssembleElementVector
  real_t AssembleElementVolume(const auto &ctx, const real_t *el_u, const real_t *elJac_d,
                               const real_t *elMetric_d, real_t *el_dudt)
  {
    const int Np_x = ctx.Np_x;
    const int Np_y = ctx.Np_y;
    const int Np_z = ctx.Np_z;
    const int dim = ctx.dim;
    const int neq = ctx.neq;

    real_t J = 0.0;
    real_t max_char_speed = 0.0;
    { // X-direction (metric row 0)
      const real_t *Dcol = ctx.Dcol; // Note: Assumes all = Np_{x,y,z}
      real_t f[MAX_EQ];
      // Zero'ing probably unnecessary: Chandrashekar flux overwrites it every time
      // for(int q = 0;q < neq;q++) f[q] = 0.0;
      for (int k = 0; k < Np_z; k++)
        for (int j = 0; j < Np_y; j++)
          for (int i = 0; i < Np_x; i++)
            {
              int id1 = k * Np_y * Np_x + j * Np_x + i;
              const real_t *state1 = el_u + id1*neq;
              J = elJac_d[id1];
              const real_t *met1 = elMetric_d+id1*dim*dim;
              for (int m = i + 1; m < Np_x; m++)
                {
                  int id2 = k * Np_y * Np_x + j * Np_x + m;
                  const real_t *state2 = el_u + id2*neq;
                  const real_t *met2 = elMetric_d + id2*dim*dim;

                  const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx, state1, state2, met1, met2, f);
                  max_char_speed = Prandtl::Device::rmax(cs, max_char_speed);

                  const real_t c1 = Dcol[m + Np_x*i];
                  const real_t c2 = Dcol[i + Np_x*m];
                  
                  real_t *rhs1 = el_dudt + id1*neq;
                  real_t *rhs2 = el_dudt + id2*neq;
                  
                  for(int q = 0;q < neq;q++)
                    {
                      rhs1[q] += c1 * f[q];
                      rhs2[q] += c2 * f[q];
                    }
                }
            }
    } // X-direction block
    
    // Y-direction (metric row 1)
    if(dim > 1) {
      // const real_t *Dcol = ctx.Dcol_y;
      // Use symmetry of TPE D operator
      const real_t *Dcol = ctx.Dcol;
      real_t g[MAX_EQ];
      for (int q = 0; q < neq; ++q) g[q] = 0;

      for (int k = 0; k < Np_z; ++k)
        for (int j = 0; j < Np_y; ++j)
          for (int i = 0; i < Np_x; ++i)
            {
              const int id1 = k*Np_y*Np_x + j*Np_x + i;
              const real_t *state1 = el_u + id1*neq;
              const real_t *met1 = elMetric_d + id1*dim*dim + 1*dim;

              for (int m = j+1; m < Np_y; ++m)
                {
                  const int id2 = k*Np_y*Np_x + m*Np_x + i;
                  const real_t *state2 = el_u + id2*neq;
                  const real_t *met2 = elMetric_d + id2*dim*dim + dim;

                  const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx, state1, state2, met1, met2, g);
                  max_char_speed = Prandtl::Device::rmax(max_char_speed, cs);

                  const real_t c1 = Dcol[m + Np_y*j]; // column j, entry m
                  const real_t c2 = Dcol[j + Np_y*m]; // column m, entry j

                  real_t *rhs1 = el_dudt + id1*neq;
                  real_t *rhs2 = el_dudt + id2*neq;

                  for (int q = 0; q < neq; ++q)
                    {
                      rhs1[q] += c1 * g[q];
                      rhs2[q] += c2 * g[q];
                    }
                }
            }
    } // Y-direction block
    
    if (dim > 2) { // Z-direction (metric row 2)
      // const real_t *Dcol = ctx.Dcol_z; // size Nz*Nz
      const real_t *Dcol = ctx.Dcol; // size Np*Np == Np_z*Np_z
      real_t h[MAX_EQ];
      for (int q = 0; q < neq; ++q) h[q] = 0;

      for (int k = 0; k < Np_z; ++k)
        for (int j = 0; j < Np_y; ++j)
          for (int i = 0; i < Np_x; ++i)
            {
              const int id1 = k*Np_y*Np_x + j*Np_x + i;
              const real_t *state1 = el_u + id1*neq;
              const real_t *met1 = elMetric_d + id1*dim*dim + 2*dim;

              for (int m = k+1; m < Nz; ++m)
                {
                  const int id2 = m*Np_y*Np_x + j*Np_x + i;
                  const real_t *state2 = el_u + id2*neq;
                  const real_t *met2 = elMetric_d + id2*dim*dim + 2*dim;

                  const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx, state1, state2, met1, met2, h);
                  max_char_speed = Prandtl::Device::rmax(max_char_speed, cs);

                  const real_t c1 = Dcol[m + Np_z*k];
                  const real_t c2 = Dcol[k + Np_z*m];

                  real_t *rhs1 = el_dudt + id1*neq;
                  real_t *rhs2 = el_dudt + id2*neq;

                  for (int q = 0; q < neq; ++q)
                    {
                      rhs1[q] += c1 * h[q];
                      rhs2[q] += c2 * h[q];
                    }
                }
            }
    } // Z-direction block
    const int NPtot = Np_x * Np_y * Np_z; // = Np_x * Np_x * Np_x (!)
    for(int id = 0;id < NPtot;id++){
      // Subcell blending off (for now)
      // const real_t invJ = (-blend_factor) / elJac_d[id];
      const real_t invJ = -1.0/elJac_d[id];
      real_t *rhs = el_dudt + id*neq;
      for(int q = 0;q < neq;q++) { rhs[q] *= invJ; }
    }

    // NOTE: Old routine saved max_char_speed as member data (ugh!)
    // This routine returns max_char_speed which should be saved
    // into an array (size = num_elements) on the caller side, and
    // then reduce/max over local elements and over ranks.
    // TODO: Fix up max_char_speed treatment on caller, and usage site
    return max_char_speed;
}

  // TODO: Update to capture max_char_speed per element
  void DGSEMNonlinearForm::Mult(const Vector &u, Vector &dudt) const
  {
    // dev: prepared DGSEMCache member
    // ctx: prepared device-safe context
    const Vector &pu = Prolongate(u);
    
    if (P)
      {
        aux2.SetSize(P->Height());
      }
    Vector &pdudt = P ? aux2 : dudt;

    mfem::Vector Ue(dev.restr_v->Height());
    mfem::Vector dUe(dev.restr_v->Height());
    dev.restr_v->Mult(pu, Ue);
    dUe = 0.0;

    const real_t *Ue_d = Ue.Read();
    real_t *dUe_d = dUe.ReadWrite();
    
    const real_t *elJac_d    = dev.elJac.Read();
    const real_t *elMetric_d = dev.elMetric.Read();
    
    const int *elem_attr_d = dev.elem_attr.Read();
    const int *attr_marker_d = dev.attr_marker.Read();

    const auto ctx = dev.ctx;
    const int dim = ctx.dim;
    const int ne = dev.ne;
    const int ndof = dev.ndof_scalar_el;
    const int neq = dev.num_eq;
    const int metric_stride = ndof * dim * dim;
    const int jac_stride    = ndof;
    const int estride = ndof*neq;
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {

      const real_t *jac_el    = elJac_d    + e * jac_stride;
      const real_t *metric_el = elMetric_d + e * metric_stride;

      const int attr = elem_attr_d[e];
      if (attr_marker_d[attr-1] == 0) { return; }

      const int eoff = e * estride;
      const real_t *u_el = Ue_d + eoff;
      real_t *du_el = dUe_d + eoff;

      const real_t cs_el = AssembleElementVolume(ctx, u_el, jac_el, metric_el, du_el);
    });

    // Scatter back to main storage
    dev.restr_v->AddMultTranspose(dUe, pdudt);
  }
}
