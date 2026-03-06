#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"
#include "LiftingScheme.hpp"
#include "ChandrashekarFlux.hpp"
#include "dgsem_cache.hpp"

namespace Prandtl
{
  
  // using namespace mfem;

  class DGSEMIntegrator : public NonlinearFormIntegrator
  {
  public:
    struct OperatorCache {
      int Np_x;
      int Np_y;
      int Np_z;
      int num_equations;
      int dim;
      int num_elements;
      int num_face_points;
      int num_interior_faces;
      mfem::Vector elJac;
      mfem::Vector elMetric;
      mfem::Vector Dhat2;
      mfem::Vector Dhat;
      mfem::Vector D;
      mfem::Vector face_normals;
      mfem::Vector face_wt_minus;
      mfem::Vector face_wt_plus;
      const mfem::FaceRestriction *restr_f = nullptr; // for face vfes (vector space)
    };
    OperatorCache cache;
  private:
    std::shared_ptr<ParMesh> pmesh;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::shared_ptr<ParGridFunction> alpha;
    NumericalFlux &rsolver;
    const NavierStokesFlux &fluxFunction;
    DenseMatrix D_T, Dhat_T, Dhat2_T;
    const int Np_x, Np_y, Np_z;
    const int num_equations, dim, num_elements;
    IntegrationRules GLIntRules;
    const IntegrationRule *ir, *ir_face, *ir_vol;
    
    real_t max_char_speed;
    real_t J, J1, J2;
    int dof, dof1, dof2;
    int id1, id2;
    int IntegrationOrder;
    
    Vector shape1, shape2;
    Vector state1, state2;
    Vector f, g, h;
    
    Vector flux_num;
    DenseMatrix flux_mat1, flux_mat2, flux_mat;
    
    DenseMatrix adj1, adj2;
    Vector metric1, metric2, met1, met2;
    Vector nor;
    
    DenseTensor F_inviscid, G_inviscid, H_inviscid;
    DenseMatrix F_viscous, G_viscous, H_viscous;

    Vector Dhat2;
    Vector D_row;
    
    Vector dU_inviscid, dU_viscous, dU_volume, dU_face1, dU_face2, dU, dU_subcell;
    
    DenseTensor SubcellMetricXi, SubcellMetricEta, SubcellMetricZeta;
    
    Array<int> alpha_indx;
    Vector el_alpha;
    
    Vector dqdx, dqdy, dqdz;
    
    Vector el_dudxi, el_dudeta, el_dudzeta;
    
    std::shared_ptr<LiftingScheme> liftingScheme;
    
    void ComputeSubcellMetrics();
    void ComputeFVFluxes(const DenseMatrix &el_u_mat, real_t alpha_value, ElementTransformation &Tr, DenseMatrix &el_dudt_mat);
#ifdef AXISYMMETRIC
    inline real_t PressureFromConservative(const Vector& U) const;
#endif
    
  public:
    
    inline real_t GetMaxCharSpeed()
    {
      return max_char_speed;
    }
    
    DGSEMIntegrator(std::shared_ptr<ParMesh> pmesh,
                    std::shared_ptr<ParFiniteElementSpace> fes0,
                    std::shared_ptr<ParGridFunction> alpha,
                    std::shared_ptr<LiftingScheme> liftingScheme,
                    NumericalFlux &rsolver, int Np);

    void CreateOperatorCache();
    void AssembleGeometricTerms();
    void AssembleElementGeometricTerms(ElementTransformation &Tr);
    void AssembleElementVectorOG(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudt);
    void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudt) override;
    void AssembleFaceVectorInviscid(const FiniteElement &el1, const FiniteElement &el2,
                                    FaceElementTransformations &Tr, const Vector &el_u,
                                    Vector &el_dudt);
    real_t AssembleElementVolumeHost(const int e, ElementTransformation &Tr, const real_t *el_u, real_t *el_dutdt);
    real_t AssembleElementVolumeHost2(const DGSEMDeviceCache &device_cache, const int e,
                                      const real_t *el_u, const real_t *jac_d,
                                      const real_t *metric_d, real_t *el_dudt);
      real_t AssembleElementVolumeDevice(const DGSEMDeviceCache &ctx,
                                       const real_t *el_u, const real_t *elJac_d,
                                       const real_t *elMetric_d, real_t *el_dudt);
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dutdt) override;
    real_t AssembleElementVectorHost(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dutdt);
    
    void AssembleFaceVector(const FiniteElement &el, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, const Vector &el_dudz, Vector &el_dudt);
    void AssembleFaceVector(const FiniteElement &el, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, Vector &el_dudt);
    void AssembleFaceVector(const FiniteElement &el, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const Vector &el_dudx, Vector &el_dudt);
    
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, const Vector &el_dudz, Vector &el_dudt);
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, Vector &el_dudt);
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, const Vector &el_dudx, Vector &el_dudt);
    
    void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz);
    void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy);
    void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx);
    
    void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz);
    void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy);
    void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx);
    ~DGSEMIntegrator() = default;
  public:

    // Originally DGSEMIntegrator.cpp::AssembleElementVector
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
      const int dof = Np_x * Np_y * Np_z;
      const int dim = ctx.dim;
      const int neq = ctx.num_equations;
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
          max_char_speed = std::max(max_char_speed,
                                    ctx.iflux.ComputeFacialFlux(ctx.gas, qMinus, qPlus,
                                                                nor_d, point_flux));
          for(int j = 0;j < neq;j++){
            rhs_face[ctx.iface_idx(0, i, j)] = wminus * point_flux[j];
            rhs_face[ctx.iface_idx(1, i, j)] = wplus * point_flux[j];
          }
        }
      // #ifdef AXISYMMETRIC
      //        Vector phys(dim);
      //        Tr.Transform(ip, phys);
      //        real_t r = phys[1]; 
      //        flux_num *= r;
      // #endif
      return max_char_speed;
    }
  };
}
