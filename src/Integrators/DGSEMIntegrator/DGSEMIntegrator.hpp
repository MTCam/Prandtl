#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"
#include "LiftingScheme.hpp"
#include "ChandrashekarFlux.hpp"

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
      mfem::Vector elJac;
      mfem::Vector elMetric;
      mfem::Vector Dhat2;
      mfem::Vector Dhat;
      mfem::Vector D;
    };
    struct DeviceCache {
      int Np_x;
      int Np_y;
      int Np_z;
      int num_equations;
      int dim;
      int num_elements;
      real_t *elJac_d;
      real_t *elMetric_d;
      real_t *Dhat2_d;
      real_t *Dhat_d;
      real_t *D_d;
      const Prandtl::Chandrashekar::InviscidFlux iflux;
      const IdealGasModel gas;
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
    void GetGeometricOperators(mfem::Vector &elJac_x, mfem::Vector &elMetric_x,
                               mfem::Vector &D, mfem::Vector &Dhat,
                               mfem::Vector &Dhat2);
    void AssembleElementVectorOG(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudt);
    void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudt) override;
    real_t AssembleElementVolumeHost(const int e, ElementTransformation &Tr, const real_t *el_u, real_t *el_dutdt);
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
  };

  namespace DGSEM {

  // Originally DGSEMIntegrator.cpp::AssembleElementVector
    MFEM_HOST_DEVICE inline
    real_t AssembleElementVolumeKernel(const DGSEMIntegrator::DeviceCache &ctx,
                                       const real_t *el_u, const real_t *elJac_d,
                                       const real_t *elMetric_d, real_t *el_dudt)
  {
    const int Np_x = ctx.Np_x;
    const int Np_y = ctx.Np_y;
    const int Np_z = ctx.Np_z;
    const int dim = ctx.dim;
    const int neq = ctx.num_equations;
    const real_t *Dhat2_d = ctx.Dhat2_d;
    //    real_t f[MAX_EQ];
    real_t f[5] = {0.,0.,0.,0.,0.};
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
              const real_t *state1 = el_u + id1*neq;
              J = elJac_d[id1];
              const real_t *met1 = elMetric_d+id1*dim*dim;
              for (int m = i + 1; m < Np_x; m++)
                {
                  int id2 = k * Np_y * Np_x + j * Np_x + m;
                  const real_t *state2 = el_u + id2*neq;
                  const real_t *met2 = elMetric_d + id2*dim*dim;

                  const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                  max_char_speed = Prandtl::Kernels::rmax(cs, max_char_speed);

                  const real_t c1 = Dhat2_d[m + Np_x*i];
                  const real_t c2 = Dhat2_d[i + Np_x*m];

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
                  // ComputeVolumeFlux *overwrites* f, so don't worry about reuse
                  const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                  max_char_speed = Prandtl::Kernels::rmax(max_char_speed, cs);

                  const real_t c1 = Dhat2_d[m + Np_y*j]; // column j, entry m
                  const real_t c2 = Dhat2_d[j + Np_y*m]; // column m, entry j

                  real_t *rhs1 = el_dudt + id1*neq;
                  real_t *rhs2 = el_dudt + id2*neq;

                  for (int q = 0; q < neq; ++q)
                    {
                      rhs1[q] += c1 * f[q];
                      rhs2[q] += c2 * f[q];
                    }
                }
            }
    } // Y-direction block
    
    if (dim > 2) { // Z-direction (metric row 2)
      for (int k = 0; k < Np_z; ++k)
        for (int j = 0; j < Np_y; ++j)
          for (int i = 0; i < Np_x; ++i)
            {
              const int id1 = k*Np_y*Np_x + j*Np_x + i;
              const real_t *state1 = el_u + id1*neq;
              const real_t *met1 = elMetric_d + id1*dim*dim + 2*dim;

              for (int m = k+1; m < Np_z; ++m)
                {
                  const int id2 = m*Np_y*Np_x + j*Np_x + i;
                  const real_t *state2 = el_u + id2*neq;
                  const real_t *met2 = elMetric_d + id2*dim*dim + 2*dim;

                  const real_t cs = ctx.iflux.ComputeVolumeFlux(ctx.gas, state1, state2, met1, met2, f);
                  max_char_speed = Prandtl::Kernels::rmax(max_char_speed, cs);

                  const real_t c1 = Dhat2_d[m + Np_z*k];
                  const real_t c2 = Dhat2_d[k + Np_z*m];

                  real_t *rhs1 = el_dudt + id1*neq;
                  real_t *rhs2 = el_dudt + id2*neq;

                  for (int q = 0; q < neq; ++q)
                    {
                      rhs1[q] += c1 * f[q];
                      rhs2[q] += c2 * f[q];
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
  }
}
