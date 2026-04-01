#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"
#include "LiftingScheme.hpp"
#include "ChandrashekarFlux.hpp"
#include "dgsem_cache_utilities.hpp"

namespace Prandtl
{
  
  class DGSEMIntegrator : public mfem::NonlinearFormIntegrator
  {
  private:
    std::shared_ptr<mfem::ParMesh> pmesh;
    std::shared_ptr<mfem::ParFiniteElementSpace> fes0;
    std::shared_ptr<mfem::ParGridFunction> alpha;
    NumericalFlux &rsolver;
    const NavierStokesFlux &fluxFunction;
    mfem::DenseMatrix D_T, Dhat_T, Dhat2_T;
    const int Np_x, Np_y, Np_z;
    const int num_equations, dim, num_elements;
    mfem::IntegrationRules GLIntRules;
    const mfem::IntegrationRule *ir, *ir_face, *ir_vol;

    real_t max_char_speed;
    real_t J, J1, J2;
    int dof, dof1, dof2;
    int id1, id2;
    int IntegrationOrder;

    mfem::Vector shape1, shape2;
    mfem::Vector state1, state2;
    mfem::Vector f, g, h;

    mfem::Vector flux_num;
    mfem::DenseMatrix flux_mat1, flux_mat2, flux_mat;

    mfem::DenseMatrix adj1, adj2;
    mfem::Vector metric1, metric2;
    mfem::Vector nor;

    mfem::DenseTensor F_inviscid, G_inviscid, H_inviscid;
    mfem::DenseMatrix F_viscous, G_viscous, H_viscous;

    mfem::Vector D_row;

    mfem::Vector dU_inviscid, dU_viscous, dU_volume, dU_face1, dU_face2, dU, dU_subcell;

    mfem::DenseTensor SubcellMetricXi, SubcellMetricEta, SubcellMetricZeta;

    mfem::Array<int> alpha_indx;
    mfem::Vector el_alpha;

    mfem::Vector dqdx, dqdy, dqdz;

    mfem::Vector el_dudxi, el_dudeta, el_dudzeta;

    std::shared_ptr<LiftingScheme> liftingScheme;
    DGSEMOperatorCache *operator_cache = nullptr;
    DGSEMDeviceCache device_cache;

    void ComputeSubcellMetrics();
    void ComputeFVFluxes(const mfem::DenseMatrix &el_u_mat, real_t alpha_value, mfem::ElementTransformation &Tr,
                         mfem::DenseMatrix &el_dudt_mat);
    void ComputeFVFluxesFromCache(const mfem::DenseMatrix &el_u_mat, real_t alpha_value,
                                  mfem::ElementTransformation &Tr, mfem::DenseMatrix &el_dudt_mat);
#ifdef AXISYMMETRIC
    inline real_t PressureFromConservative(const mfem::Vector& U) const;
#endif

  public:
    void SetOperatorCache(DGSEMOperatorCache *cache_)
    { operator_cache = cache_;
      GetDeviceCache(*operator_cache, device_cache);
    };
    inline real_t GetMaxCharSpeed()
    {
      return max_char_speed;
    }

    DGSEMIntegrator(std::shared_ptr<mfem::ParMesh> pmesh,
                    std::shared_ptr<mfem::ParFiniteElementSpace> fes0,
                    std::shared_ptr<mfem::ParGridFunction> alpha,
                    std::shared_ptr<LiftingScheme> liftingScheme,
                    NumericalFlux &rsolver, int Np);

    void AssembleFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u,
                            mfem::Vector &el_dudt) override;
    void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                               const mfem::Vector &el_u, mfem::Vector &el_dutdt) override;
    
    void AssembleFaceVector(const mfem::FiniteElement &el, const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u,
                            const mfem::Vector &el_dudx, const mfem::Vector &el_dudy,
                            const mfem::Vector &el_dudz, mfem::Vector &el_dudt);
    void AssembleFaceVector(const mfem::FiniteElement &el, const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u,
                            const mfem::Vector &el_dudx, const mfem::Vector &el_dudy, mfem::Vector &el_dudt);
    void AssembleFaceVector(const mfem::FiniteElement &el, const mfem::FiniteElement &el2, mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u, const mfem::Vector &el_dudx, mfem::Vector &el_dudt);

    void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &el_u, const mfem::Vector &el_dudx, const mfem::Vector &el_dudy, const mfem::Vector &el_dudz, mfem::Vector &el_dudt);
    void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &el_u, const mfem::Vector &el_dudx, const mfem::Vector &el_dudy, mfem::Vector &el_dudt);
    void AssembleElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr, const mfem::Vector &el_u, const mfem::Vector &el_dudx, mfem::Vector &el_dudt);

    void AssembleLiftingFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
                                   mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u,
                                   mfem::Vector &el_dudx, mfem::Vector &el_dudy, mfem::Vector &el_dudz);
    void AssembleLiftingFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
                                   mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u,
                                   mfem::Vector &el_dudx, mfem::Vector &el_dudy);
    void AssembleLiftingFaceVector(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
                                   mfem::FaceElementTransformations &Tr, const mfem::Vector &el_u,
                                   mfem::Vector &el_dudx);
    
    void AssembleLiftingElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                                      const mfem::Vector &el_u, mfem::Vector &el_dudx, mfem::Vector &el_dudy,
                                      mfem::Vector &el_dudz);
    void AssembleLiftingElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                                      const mfem::Vector &el_u, mfem::Vector &el_dudx, mfem::Vector &el_dudy);
    void AssembleLiftingElementVector(const mfem::FiniteElement &el, mfem::ElementTransformation &Tr,
                                      const mfem::Vector &el_u, mfem::Vector &el_dudx);
    ~DGSEMIntegrator() = default;
  public:

    template<typename ContextType>
    MFEM_HOST_DEVICE inline
    static real_t AssembleElementVolumeKernel(const ContextType &ctx,
                                              const real_t *el_u, const real_t *elJac_d,
                                              const real_t *elMetric_d, real_t *el_dudt)
    {

      // TODO: bring subcell blending back SUBCELL_FV_BLENDING
      // TODO: bring back axisymmetric terms
      const int Np_x = ctx.Np_x;
      const int Np_y = ctx.Np_y;
      const int Np_z = ctx.Np_z;
      const int dim = ctx.dim;
      const int neq = ctx.num_equations;
      const int dof = Np_x * Np_y * Np_z;
      const real_t *Dhat2_d = ctx.Dhat2_d;

      // TODO: Really integrate/use MAX_EQ or equivalent
      //    real_t f[MAX_EQ];
      real_t f[5] = {0.,0.,0.,0.,0.};
      real_t state1[5];
      real_t state2[5];
      real_t J = 0.0;
      real_t max_char_speed = 0.0;
      { // X-direction (metric row 0)
        // Zero'ing probably unnecessary: Chandrashekar flux overwrites it every time
        // for(int q = 0;q < neq;q++) f[q] = 0.0;
        for (int k = 0; k < Np_z; k++)
          for (int j = 0; j < Np_y; j++)
            for (int i = 0; i < Np_x; i++)
              {
                int id1 = k * Np_y * Np_x + j * Np_x + i;
                Kernels::el_gather_state(el_u, dof, neq, id1, state1);
                J = elJac_d[id1];
                const real_t *met1 = elMetric_d+id1*dim*dim;
                for (int m = i + 1; m < Np_x; m++)
                  {
                    int id2 = k * Np_y * Np_x + j * Np_x + m;
                    Kernels::el_gather_state(el_u, dof, neq, id2, state2);
                    const real_t *met2 = elMetric_d + id2*dim*dim;
                  
                    const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                    max_char_speed = Kernels::rmax(cs, max_char_speed);
                  
                    const real_t c1 = Dhat2_d[m + Np_x*i];
                    const real_t c2 = Dhat2_d[i + Np_x*m];
                    Kernels::el_scatter_add(f, dof, neq, id1, c1, el_dudt);
                    Kernels::el_scatter_add(f, dof, neq, id2, c2, el_dudt);
                  
                  }
              }
      } // X-direction block
    
      // Y-direction (metric row 1)
      if(dim > 1) {
        for (int k = 0; k < Np_z; ++k)
          for (int j = 0; j < Np_y; ++j)
            for (int i = 0; i < Np_x; ++i)
              {
                const int id1 = k*Np_y*Np_x + j*Np_x + i;
                Kernels::el_gather_state(el_u, dof, neq, id1, state1);
                const real_t *met1 = elMetric_d + id1*dim*dim + 1*dim;
              
                for (int m = j+1; m < Np_y; ++m)
                  {
                    const int id2 = k*Np_y*Np_x + m*Np_x + i;
                    Kernels::el_gather_state(el_u, dof, neq, id2, state2);
                    const real_t *met2 = elMetric_d + id2*dim*dim + dim;
                    // ComputeVolumeFlux *overwrites* f, so don't worry about reuse
                    const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                    max_char_speed = Kernels::rmax(max_char_speed, cs);
                  
                    const real_t c1 = Dhat2_d[m + Np_y*j]; // column j, entry m
                    const real_t c2 = Dhat2_d[j + Np_y*m]; // column m, entry j
                    Kernels::el_scatter_add(f, dof, neq, id1, c1, el_dudt);
                    Kernels::el_scatter_add(f, dof, neq, id2, c2, el_dudt);
                  
                  }
              }
      } // Y-direction block
    
      if (dim > 2) { // Z-direction (metric row 2)
        for (int k = 0; k < Np_z; ++k)
          for (int j = 0; j < Np_y; ++j)
            for (int i = 0; i < Np_x; ++i)
              {
                const int id1 = k*Np_y*Np_x + j*Np_x + i;
                Kernels::el_gather_state(el_u, dof, neq, id1, state1);
                const real_t *met1 = elMetric_d + id1*dim*dim + 2*dim;
              
                for (int m = k+1; m < Np_z; ++m)
                  {
                    const int id2 = m*Np_y*Np_x + j*Np_x + i;
                    Kernels::el_gather_state(el_u, dof, neq, id2, state2);
                    const real_t *met2 = elMetric_d + id2*dim*dim + 2*dim;
                  
                    const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                    max_char_speed = Kernels::rmax(max_char_speed, cs);
                  
                    const real_t c1 = Dhat2_d[m + Np_z*k];
                    const real_t c2 = Dhat2_d[k + Np_z*m];
                    Kernels::el_scatter_add(f, dof, neq, id1, c1, el_dudt);
                    Kernels::el_scatter_add(f, dof, neq, id2, c2, el_dudt);
                  }
              }
      } // Z-direction block
      // const int NPtot = Np_x * Np_y * Np_z; // = Np_x * Np_x * Np_x (!)
      Kernels::el_scale(elJac_d, -1.0, dof, neq, el_dudt);
      // for(int id = 0;id < NPtot;id++){
      //   // Subcell blending off (for now)
      //   // const real_t invJ = (-blend_factor) / elJac_d[id];
      //   const real_t invJ = -1.0/elJac_d[id];
      //   for(int q = 0;q < neq;q++) { el_dudt[id + q*NPtot] *= invJ; }
      // }
    
      // NOTE: Old routine saved max_char_speed as member data (ugh!)
      // This routine returns max_char_speed which should be saved
      // into an array (size = num_elements) on the caller side, and
      // then reduce/max over local elements and over ranks.
      // TODO: Fix up max_char_speed treatment on caller, and usage site
      return max_char_speed;
    }

    template<typename ContextT>
    MFEM_HOST_DEVICE static real_t AssembleElementFaceKernel(const ContextT &ctx, const real_t *u_face,
                                                             const real_t *nor_face,const real_t *w_minus,
                                                             const real_t *w_plus, real_t *rhs_face)
    { // TODO: Fix hard-coded sizes (5)
      real_t max_char_speed = 0.0;
      real_t point_flux[5];
      real_t qMinus[5];
      real_t qPlus[5];
      const int nfp = ctx.num_face_points;
      const int neq = ctx.num_equations;
      const int dim = ctx.dim;
      // auto idx = [=](int side, int fp, int eq) -> int
      // {
      //   return (((side)*neq + eq)*nfp + fp);
      // };
      for (int i = 0; i < nfp; i++)
        {
          const real_t *nor_d = nor_face + i*dim;
          const real_t wminus = -w_minus[i];
          const real_t wplus = w_plus[i];
          // Could avoid these copy-in,out 
          for(int j = 0;j < neq;j++){
            qMinus[j] = u_face[ctx.iface_idx(0,i,j)];
            qPlus[j] = u_face[ctx.iface_idx(1,i,j)];
          }
          max_char_speed = \
            Kernels::rmax(max_char_speed, ctx.iflux.ComputeFaceFlux(ctx.gas, qMinus, qPlus,
                                                                    nor_d, point_flux));
          for(int j = 0;j < neq;j++){
            rhs_face[ctx.iface_idx(0, i, j)] = wminus * point_flux[j];
            rhs_face[ctx.iface_idx(1, i, j)] = wplus * point_flux[j];
          }
        }
      // #ifdef AXISYMMETRIC
      //        mfem::Vector phys(dim);
      //        Tr.Transform(ip, phys);
      //        real_t r = phys[1]; 
      //        flux_num *= r;
      // #endif
      return max_char_speed;
    }
  };

  template<typename ContextT>
  MFEM_HOST_DEVICE
  inline void AssembleElementFV(const ContextT &ctx, const real_t *el_u, const real_t *elJac,
                                const real_t *el_metric_xi, const real_t *el_metric_eta,
                                const real_t *el_metric_zeta, const real_t alpha, real_t *el_dudt)
  {

    MFEM_ASSERT(ctx.num_equations <= Prandtl::MAXEQ,        
                "num_equations exceeds Prandtl::MAXEQ");
    const int dim = ctx.dim;
    const int Np_x = ctx.Np_x;
    const int Np_y = ctx.Np_y;
    const int Np_z = ctx.Np_z;
    const int npe = Np_x * Np_y * Np_z;
    const int neq = ctx.num_equations;

    const real_t *qWgt = ctx.subcell_weights_d;

    real_t flux_num[Prandtl::MAXEQ];
    real_t du_subcell[Prandtl::MAXEQ];
    real_t state1[Prandtl::MAXEQ];
    real_t state2[Prandtl::MAXEQ];
 
    for (int ii = 0; ii < Prandtl::MAXEQ; ++ii) {
      flux_num[ii] = NAN;
      du_subcell[ii] = NAN;
      state1[ii] = NAN;
      state2[ii] = NAN;
    }

    real_t ws = 0.0;

    for (int k = 0; k < Np_z; k++)
      {
        for (int j = 0; j < Np_y; j++)
          {
            for(int ii=0;ii < neq;ii++)
              du_subcell[ii] = 0.0;
            int id1 = k * Np_y * Np_x + j * Np_x;
            Kernels::el_gather_state(el_u, npe, neq, id1, state1);

            for (int i = 0; i < Np_x - 1; i++)
              {
                const int id2 = id1 + 1;
                Kernels::el_gather_state(el_u, npe, neq, id2, state2);
                MFEM_ASSERT(id2 < npe_metric_xi,   "xi metric index OOB");
                MFEM_ASSERT(id2 < npe_metric_eta,  "eta metric index OOB");
                MFEM_ASSERT(id2 < npe_metric_zeta, "zeta metric index OOB");
                const real_t *nor = el_metric_xi + id2*dim;

                real_t pws = ctx.iflux.ComputeFaceFlux(ctx.gas, state1, state2 , nor, flux_num);
                ws = Kernels::rmax(ws, pws);

                const real_t fac = 1.0 / (elJac[id1] * qWgt[i]);
                for(int ii = 0;ii < neq;ii++)
                  du_subcell[ii] = (du_subcell[ii] - flux_num[ii])*fac;

                Kernels::el_scatter_assign(du_subcell, npe, neq, id1, 1.0, el_dudt);

                for(int ii = 0;ii < neq;ii++){
                  du_subcell[ii] = flux_num[ii];
                  state1[ii] = state2[ii];
                }
                id1 = id2;
              }

            const real_t fac = 1.0 / (elJac[id1] * qWgt[Np_x - 1]);
            Kernels::el_scatter_assign(du_subcell, npe, neq, id1, fac, el_dudt);
          }
      }

    if (dim > 1)
      {
        for (int k = 0; k < Np_z; k++)
          {
            for (int i = 0; i < Np_x; i++)
              {
                for(int ii = 0;ii < neq;ii++)
                  du_subcell[ii] = 0.0;
                int id1 = k * Np_y * Np_x + i;
                Kernels::el_gather_state(el_u, npe, neq, id1, state1);

                for (int j = 0; j < Np_y - 1; j++)
                  {
                    const int id2 = k * Np_y * Np_x + (j + 1) * Np_x + i;
                    
                    Kernels::el_gather_state(el_u, npe, neq, id2, state2);
                    MFEM_ASSERT(id2 < npe_metric_xi,   "xi metric index OOB");
                    MFEM_ASSERT(id2 < npe_metric_eta,  "eta metric index OOB");
                    MFEM_ASSERT(id2 < npe_metric_zeta, "zeta metric index OOB");
                    const real_t *nor = el_metric_eta + id2*dim;

                    real_t pws = ctx.iflux.ComputeFaceFlux(ctx.gas, state1, state2, nor, flux_num);
                    ws = Kernels::rmax(ws, pws);

                    real_t fac = 1.0 / (elJac[id1] * qWgt[j]);
                    for(int ii = 0;ii < neq;ii++)
                      du_subcell[ii] = (du_subcell[ii] - flux_num[ii])*fac;

                    Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);

                    for(int ii = 0;ii < neq;ii++){
                      du_subcell[ii] = flux_num[ii];
                      state1[ii] = state2[ii];
                    }
                    id1 = id2;                   
                  }

                real_t fac = 1.0 / (elJac[id1] * qWgt[Np_y - 1]);
                Kernels::el_scatter_add(du_subcell, npe, neq, id1, fac, el_dudt);
              }
          }
        if (dim > 2)
          {
            for (int j = 0; j < Np_y; j++)
              {
                for (int i = 0; i < Np_x; i++)
                  {

                    for(int ii = 0;ii < neq;ii++)
                      du_subcell[ii] = 0.0;
                    int id1 = j * Np_x + i;
                    Kernels::el_gather_state(el_u, npe, neq, id1, state1);

                    for (int k = 0; k < Np_z - 1; k++)
                      {
                        const int id2 = (k + 1) * Np_y * Np_x + j * Np_x + i;

                        Kernels::el_gather_state(el_u, npe, neq, id2, state2);
                        const real_t *nor = el_metric_zeta + id2*dim;

                        real_t pws = ctx.iflux.ComputeFaceFlux(ctx.gas, state1, state2, nor, flux_num); 
                        ws = Kernels::rmax(ws, pws);

                        real_t fac = 1.0 / (elJac[id1] * qWgt[k]);
                        for(int ii = 0;ii < neq;ii++)
                          du_subcell[ii] = (du_subcell[ii] - flux_num[ii])*fac;

                        Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);

                        for(int ii = 0;ii < neq;ii++){
                          du_subcell[ii] = flux_num[ii];
                          state1[ii] = state2[ii];
                        }
                        id1 = id2;            
                      }

                    real_t fac = 1.0 / (elJac[id1] * qWgt[Np_z - 1]);
                    Kernels::el_scatter_add(du_subcell, npe, neq, id1, fac, el_dudt);
                  }
              }
          }
      }
    const int npoints = npe * neq;
    for(int ipt = 0;ipt < npoints;ipt++){
      el_dudt[ipt] *= alpha;
    }
    // return max_char_speed;
  };
  
  template<typename ContextT>
  MFEM_HOST_DEVICE inline real_t ComputeFVFluxesKernelIsh(const ContextT &device_cache,
                                                          const int e,
                                                          const real_t *el_u,const real_t alpha_value,
                                                          real_t *el_dudt)
  {
    const int dim = device_cache.dim;
    const int Np_x = device_cache.Np_x;
    const int Np_y = device_cache.Np_y;
    const int Np_z = device_cache.Np_z;
    const int neq = device_cache.num_equations;
    const int npe = Np_x * Np_y * Np_z;
    const int ndofe = npe * neq;
    MFEM_ASSERT(dim == 2, "Bad dim");
    MFEM_ASSERT(neq == (dim+2), "Bad neq");
    MFEM_ASSERT(npe == 16, "Bad npe");
    // We have only isotropic TPE so actually
    // Np_{y,z} should == Np_x || 1, but for clarity:
    const int npe_metric_xi = (Np_x + 1)*Np_y*Np_z;
    const int npe_metric_eta = Np_x*(Np_y + 1)*Np_z;
    const int npe_metric_zeta = Np_x * Np_y * (Np_z + 1);

    const real_t *el_metric_xi = device_cache.subcell_metric_xi_d + e*npe_metric_xi*dim;
    const real_t *el_metric_eta = (dim > 1 ? device_cache.subcell_metric_eta_d + e*npe_metric_eta*dim :
                                   nullptr);
    const real_t *el_metric_zeta = (dim > 2 ? device_cache.subcell_metric_zeta_d + e*npe_metric_zeta*dim :
                                    nullptr);

    const real_t *elJac = device_cache.elJac_d + e*npe;
    const real_t *qWgt = device_cache.subcell_weights_d;

    real_t max_char_speed = 0.0;
    real_t flux_num[5];
    real_t du_subcell[5];
    real_t state1_local[5];
    real_t state2_local[5];

    for (int k = 0; k < Np_z; k++)
      {
        for (int j = 0; j < Np_y; j++)
          {
            for(int q = 0; q < neq;q++){
              du_subcell[q] = 0.0;
            }
            int id1 = k * Np_y * Np_x + j * Np_x;
            Kernels::el_gather_state(el_u, npe, neq, id1, state1_local);
            for (int i = 0; i < Np_x - 1; i++)
              {
                int id2 = id1 + 1;
                Kernels::el_gather_state(el_u, npe, neq, id2, state2_local);
                const real_t *nor = el_metric_xi + id2*dim;

                max_char_speed = \
                  std::max(max_char_speed,
                           device_cache.iflux.ComputeFaceFlux(device_cache.gas, state1_local,
                                                              state2_local, nor, flux_num));
                for(int q = 0; q < neq;q++){
                  du_subcell[q] -= flux_num[q];
                }
                for(int q = 0; q < neq;q++){
                  du_subcell[q] /= (elJac[id1] * qWgt[i]);
                }
                Kernels::el_scatter_assign(du_subcell, npe, neq, id1, 1.0, el_dudt);
                for(int q = 0; q < neq;q++){
                  du_subcell[q] = flux_num[q];
                }
                for(int q = 0;q < neq;q++){
                  state1_local[q] = state2_local[q];
                }
                id1 = id2;
              }
            for(int q = 0;q < neq;q++){
              du_subcell[q] /= (elJac[id1] * qWgt[Np_x-1]);
            }
            Kernels::el_scatter_assign(du_subcell, npe, neq, id1, 1.0, el_dudt);
          }
      }
  
    if (dim > 1)
      {
        for (int k = 0; k < Np_z; k++)
          {
            for (int i = 0; i < Np_x; i++)
              {
                for(int q = 0; q < neq;q++){
                  du_subcell[q] = 0.0;
                }
                int id1 = k * Np_y * Np_x + i;
                Kernels::el_gather_state(el_u, npe, neq, id1,
                                         state1_local);
                for (int j = 0; j < Np_y - 1; j++)
                  {
                    int id2 = k * Np_y * Np_x + (j + 1) * Np_x + i;
                    Kernels::el_gather_state(el_u, npe, neq, id2,
                                             state2_local);
                    const real_t *nor = el_metric_eta + id2*dim;
                    max_char_speed = \
                      std::max(max_char_speed,
                               device_cache.iflux.ComputeFaceFlux(device_cache.gas,
                                                                  state1_local,
                                                                  state2_local,
                                                                  nor, flux_num));
                  
                    for(int q = 0;q < neq;q++){
                      du_subcell[q] -= flux_num[q];
                    }
                    for(int q = 0;q < neq;q++){
                      du_subcell[q] /= (elJac[id1] * qWgt[j]);
                    }
                    Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                    for(int q = 0;q < neq;q++){
                      du_subcell[q] = flux_num[q];
                      state1_local[q] = state2_local[q];
                    }
                    id1 = id2;                   
                  }
                for(int q = 0;q < neq;q++){
                  du_subcell[q] /= (elJac[id1] * qWgt[Np_y - 1]);
                }
                Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
              }
          }
        if (dim > 2)
          {
            const real_t *metric_zeta = device_cache.subcell_metric_zeta_d + e*npe_metric_zeta*dim;
            for (int j = 0; j < Np_y; j++)
              {
                for (int i = 0; i < Np_x; i++)
                  {
                    for(int q = 0; q < neq;q++){
                      du_subcell[q] = 0.0;
                    }
                    int id1 = j * Np_x + i;
                    Kernels::el_gather_state(el_u, npe, neq, id1,
                                             state1_local);
                    for (int k = 0; k < Np_z - 1; k++)
                      {
                        int id2 = (k + 1) * Np_y * Np_x + j * Np_x + i;
                        Kernels::el_gather_state(el_u, npe, neq, id2,
                                                 state2_local);
                        const real_t *nor = el_metric_zeta + id2*dim; 
                        max_char_speed = \
                          std::max(max_char_speed,
                                   device_cache.iflux.ComputeFaceFlux(device_cache.gas, state1_local,
                                                                      state2_local, nor, flux_num));
                        for(int q = 0;q < neq;q++){
                          du_subcell[q] -= flux_num[q];
                        }
                        for(int q = 0;q < neq;q++){
                          du_subcell[q] /= (elJac[id1] * qWgt[k]);
                        }
                        Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                      
                        for(int q = 0;q < neq;q++){
                          du_subcell[q] = flux_num[q];
                          state1_local[q] = state2_local[q];
                        }
                        id1 = id2;            
                      }
                    for(int q = 0;q < neq;q++){
                      du_subcell[q] /= (elJac[id1] * qWgt[Np_z - 1]);
                    }
                    Kernels::el_scatter_add(du_subcell, npe, neq, id1, 1.0, el_dudt);
                  }
              }
          }
      }
    for(int q = 0;q < neq*npe;q++){
      el_dudt[q] *= alpha_value;
    }
    return max_char_speed;
  };
}
