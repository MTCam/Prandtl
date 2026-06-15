#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"

namespace Prandtl
{

using namespace mfem;

class BdrFaceIntegrator;

class LiftingScheme
{
protected:
    DenseMatrix grad_mat1, grad_mat2, D_T, adj;
    Vector state1, state2, grad_state, f, g, h, dU, nor, shape1, shape2;
    Vector el_dudxi, el_dudeta, el_dudzeta;
    const IntegrationRule *ir, *ir_face, *ir_vol;
    real_t J, J1, J2;
    int id, dof, dof1, dof2, num_equations, Np_x, Np_y, Np_z;

    LiftingScheme() = default;
public:
    virtual void SetLiftingParameters(const IntegrationRule *ir, const IntegrationRule *ir_face, const IntegrationRule *ir_vol, const DenseMatrix &D_T, int num_equations, int Np, int dim);
    virtual ~LiftingScheme() = default;

    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz) = 0;
    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy) = 0;
    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx) = 0;
    
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz) = 0;
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy) = 0;
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx) = 0;

    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz) = 0;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables, Vector &el_dudx, Vector &el_dudy) = 0;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables, Vector &el_dudx) = 0;
};
    
} // namespace Prandtl

