#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"
#include "LiftingScheme.hpp"

namespace Prandtl
{

using namespace mfem;

class DGSEMIntegrator : public NonlinearFormIntegrator
{
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
    Vector metric1, metric2;
    Vector nor;

    DenseTensor F_inviscid, G_inviscid, H_inviscid;
    DenseMatrix F_viscous, G_viscous, H_viscous;

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

    void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudt) override;
    void AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dutdt) override;
    
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

}
