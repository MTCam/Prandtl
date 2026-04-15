#pragma once

#include "LiftingScheme.hpp"
#include "prandtl_kernels.hpp"
#include "bc_kernels.hpp"

namespace Prandtl
{

using namespace mfem;

class LiftingBR1 : public LiftingScheme
{
public:
    LiftingBR1();

    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz) override;
    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u, Vector &el_dudx, Vector &el_dudy) override;
    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u, Vector &el_dudx) override;

    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx,
                                              Vector &el_dudy, Vector &el_dudz) override;
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr,
                                              const Vector &el_u, Vector &el_dudx, Vector &el_dudy) override;
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx) override;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx,
                                              Vector &el_dudy, Vector &el_dudz) override;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy) override;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx) override;
  template <typename ContextType>
  MFEM_HOST_DEVICE inline
  static void AssembleGradElementVolumeKernel2D(const ContextType &ctx,
                                                const real_t *el_u,
                                                const real_t *elJac_d,
                                                const real_t *elMetric_d,
                                                real_t *el_dudx,
                                                real_t *el_dudy)
  {
    const int Np_x = ctx.Np_x;
    const int Np_y = ctx.Np_y;
    const int neq  = ctx.num_equations;
    const int dim  = ctx.dim;
    const int dof  = Np_x * Np_y;
    const real_t *D_d = ctx.D_d;

    // Keep MAX_EQ in mind later if neq can exceed 5.
    real_t dudxi[Prandtl::MAXEQ];
    real_t dudeta[Prandtl::MAXEQ];

    for (int j = 0; j < Np_y; ++j)
      {
        for (int i = 0; i < Np_x; ++i)
          {
            const int id = j * Np_x + i;

            for (int q = 0; q < neq; ++q)
              {
                dudxi[q]  = 0.0;
                dudeta[q] = 0.0;
              }
            
            // Reference-space derivatives.
            for (int l = 0; l < Np_x; ++l)
              {
                const int id_x = j * Np_x + l;
                const int id_y = l * Np_x + i;
                
                const real_t c_xi  = D_d[l + Np_x * i]; // legacy D_T(l,i)
                const real_t c_eta = D_d[l + Np_x * j]; // legacy D_T(l,j)
                
                for (int q = 0; q < neq; ++q)
                  {
                    dudxi[q]  += el_u[id_x + q * dof] * c_xi;
                    dudeta[q] += el_u[id_y + q * dof] * c_eta;
                  }
              }

            const real_t invJ = 1.0 / elJac_d[id];
            const real_t *adj = elMetric_d + id * dim * dim;

            // adj stored row-major per point:
            // [ adj[0] adj[1] ]
            // [ adj[2] adj[3] ]
            for (int q = 0; q < neq; ++q)
              {
                el_dudx[id + q * dof] = invJ * (dudxi[q] * adj[0] + dudeta[q] * adj[2]);
                el_dudy[id + q * dof] = invJ * (dudxi[q] * adj[1] + dudeta[q] * adj[3]);
              }
          }
      }
  }

  template <typename ContextT>
  MFEM_HOST_DEVICE inline
  static void AssembleGradInteriorFaceKernel2D(const ContextT &ctx,
                                               const real_t *u_face,
                                               const real_t *nor_face,
                                               const real_t *w_minus,
                                               const real_t *w_plus,
                                               real_t *rhs_face_x,
                                               real_t *rhs_face_y)
  {
    const int nfp = ctx.num_face_points;
    const int neq = ctx.num_equations;
    const int dim = ctx.dim;
    
    real_t qMinus[Prandtl::MAXEQ];
    real_t qPlus[Prandtl::MAXEQ];
    real_t jump[Prandtl::MAXEQ];
    
    for (int i = 0; i < nfp; ++i)
      {
        const real_t *nor_d = nor_face + i * dim;
        const real_t nx = nor_d[0];
        const real_t ny = nor_d[1];
        
        const real_t wminus = w_minus[i];
        const real_t wplus  = w_plus[i];

        for (int q = 0; q < neq; ++q)
          {
            qMinus[q] = u_face[ctx.iface_idx(0, i, q)];
            qPlus[q]  = u_face[ctx.iface_idx(1, i, q)];
            jump[q]   = real_t(0.5) * (qPlus[q] - qMinus[q]);
          }
        
        for (int q = 0; q < neq; ++q)
          {
            const real_t fx = jump[q] * nx;
            const real_t fy = jump[q] * ny;
            
            rhs_face_x[ctx.iface_idx(0, i, q)] = wminus * fx;
            rhs_face_x[ctx.iface_idx(1, i, q)] = wplus  * fx;
            
            rhs_face_y[ctx.iface_idx(0, i, q)] = wminus * fy;
            rhs_face_y[ctx.iface_idx(1, i, q)] = wplus  * fy;
          }
      }
  }
  

  template <typename DeviceCacheT>
  MFEM_HOST_DEVICE inline
  static void AssembleGradBoundaryPointKernel2D(const DeviceCacheT &dc,
                                                const Prandtl::BCDescriptor &bc,
                                                const real_t *u_face,
                                                const real_t *nor_point,
                                                const real_t scale,
                                                const int fp,
                                                real_t *rhs_face_x,
                                                real_t *rhs_face_y)
  {
    const int nfp = dc.num_face_points;
    const int neq = dc.num_equations;
    
    real_t state1[Prandtl::MAXEQ];
    real_t fluxN[Prandtl::MAXEQ];
    real_t fx[Prandtl::MAXEQ];
    real_t fy[Prandtl::MAXEQ];
    
    Prandtl::Kernels::el_gather_state(u_face, nfp, neq, fp, state1);
    
    Prandtl::BC::ComputeBdrFaceGradFlux(dc, bc, state1, fluxN);
    
    const real_t nx = nor_point[0];
    const real_t ny = nor_point[1];
    
    for (int q = 0; q < neq; ++q)
      {
        fx[q] = fluxN[q] * nx;
        fy[q] = fluxN[q] * ny;
      }
    
    Prandtl::Kernels::el_scatter_add(fx, nfp, neq, fp, scale, rhs_face_x);
    Prandtl::Kernels::el_scatter_add(fy, nfp, neq, fp, scale, rhs_face_y);
  }

};
  
}
