#include "LiftingBR1.hpp"
#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{

LiftingBR1::LiftingBR1()
    : LiftingScheme() {}

void LiftingBR1::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz)
{
    // const int dof1 = el1.GetDof();
    // const int dof2 = el2.GetDof();

    // shape1.SetSize(dof1);
    // shape2.SetSize(dof2);

    el_dudx.SetSize((dof1 + dof2) * num_equations);
    el_dudy.SetSize((dof1 + dof2) * num_equations);
    el_dudz.SetSize((dof1 + dof2) * num_equations);
    el_dudx = 0.0;
    el_dudy = 0.0;
    el_dudz = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    DenseMatrix el_dudx_mat2(el_dudx.GetData() + dof1 * num_equations, dof2, num_equations);

    DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
    DenseMatrix el_dudy_mat2(el_dudy.GetData() + dof1 * num_equations, dof2, num_equations);

    DenseMatrix el_dudz_mat1(el_dudz.GetData(), dof1, num_equations);
    DenseMatrix el_dudz_mat2(el_dudz.GetData() + dof1 * num_equations, dof2, num_equations);

    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir_face->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        J1 = Tr.GetElement1Transformation().Weight();
        J2 = Tr.GetElement2Transformation().Weight();
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

        el_u_mat1.MultTranspose(shape1, state1);
        el_u_mat2.MultTranspose(shape2, state2);
   
        CalcOrtho(Tr.Jacobian(), nor);

        subtract(0.5, state2, state1, f);
        h = g = f;

        f *= nor(0);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, f, el_dudx_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, f, el_dudx_mat2);

        g *= nor(1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, g, el_dudy_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, g, el_dudy_mat2);

        h *= nor(2);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, h, el_dudz_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, h, el_dudz_mat2);       
    }
}

void LiftingBR1::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy)
{
    // const int dof1 = el1.GetDof();
    // const int dof2 = el2.GetDof();

    // shape1.SetSize(dof1);
    // shape2.SetSize(dof2);

    el_dudx.SetSize((dof1 + dof2) * num_equations);
    el_dudy.SetSize((dof1 + dof2) * num_equations);
    el_dudx = 0.0;
    el_dudy = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    DenseMatrix el_dudx_mat2(el_dudx.GetData() + dof1 * num_equations, dof2, num_equations);

    DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
    DenseMatrix el_dudy_mat2(el_dudy.GetData() + dof1 * num_equations, dof2, num_equations);

    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir_face->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        J1 = Tr.GetElement1Transformation().Weight();
        J2 = Tr.GetElement2Transformation().Weight();
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

        el_u_mat1.MultTranspose(shape1, state1);
        el_u_mat2.MultTranspose(shape2, state2);
   
        CalcOrtho(Tr.Jacobian(), nor);

        subtract(0.5, state2, state1, f);
        g = f;
        f *= nor(0);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, f, el_dudx_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, f, el_dudx_mat2);

        g *= nor(1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, g, el_dudy_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, g, el_dudy_mat2);
    }
}

void LiftingBR1::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx)
{
    // const int dof1 = el1.GetDof();
    // const int dof2 = el2.GetDof();

    // shape1.SetSize(dof1);
    // shape2.SetSize(dof2);

    el_dudx.SetSize((dof1 + dof2) * num_equations);
    el_dudx = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    DenseMatrix el_dudx_mat2(el_dudx.GetData() + dof1 * num_equations, dof2, num_equations);

    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir_face->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        J1 = Tr.GetElement1Transformation().Weight();
        J2 = Tr.GetElement2Transformation().Weight();
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

        el_u_mat1.MultTranspose(shape1, state1);
        el_u_mat2.MultTranspose(shape2, state2);
   
        nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;

        subtract(0.5, state2, state1, f);

        f *= nor(0);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, f, el_dudx_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, f, el_dudx_mat2);
    }
}

void LiftingBR1::AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz)
{
    // const int dof = el.GetDof();

    el_dudx.SetSize(dof * num_equations);
    el_dudy.SetSize(dof * num_equations);
    el_dudz.SetSize(dof * num_equations);
    el_dudx = 0.0;
    el_dudy = 0.0;
    el_dudz = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    DenseMatrix el_dudx_mat(el_dudx.GetData(), dof, num_equations);
    DenseMatrix el_dudy_mat(el_dudy.GetData(), dof, num_equations);
    DenseMatrix el_dudz_mat(el_dudz.GetData(), dof, num_equations);

    grad_mat1.GetColumnReference(0, el_dudxi);
    grad_mat1.GetColumnReference(1, el_dudeta);
    grad_mat1.GetColumnReference(2, el_dudzeta);

    for (int k = 0; k < Np_x; k++)
    {
        for (int j = 0; j < Np_y; j++)
        {
            for (int i = 0; i < Np_z; i++)
            {
                id = k * Np_x * Np_y + j * Np_x + i;
                const IntegrationPoint &ip1 = ir_vol->IntPoint(id);
                Tr.SetIntPoint(&ip1);
                J = Tr.Weight();
                adj = Tr.AdjugateJacobian();

                grad_mat1 = 0.0;

                for (int l = 0; l < Np_x; l++)
                {
                    el_u_mat.GetRow(k * Np_y * Np_x + j * Np_x + l, dU);
                    dU *= D_T(l, i);
                    el_dudxi += dU;

                    el_u_mat.GetRow(k * Np_y * Np_x + l * Np_x + i, dU);
                    dU *= D_T(l, j);
                    el_dudeta += dU;

                    el_u_mat.GetRow(l * Np_y * Np_x + j * Np_x + i, dU);
                    dU *= D_T(l, k);
                    el_dudzeta += dU;
                }

                mfem::Mult(grad_mat1, adj, grad_mat2);
                grad_mat2 *= 1.0 / J;

                grad_mat2.GetColumn(0, grad_state);
                el_dudx_mat.SetRow(id, grad_state);

                grad_mat2.GetColumn(1, grad_state);
                el_dudy_mat.SetRow(id, grad_state);

                grad_mat2.GetColumn(2, grad_state);
                el_dudz_mat.SetRow(id, grad_state);
            }
        }
    }
}


void LiftingBR1::AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr,
                                              const Vector &el_u, Vector &el_dudx, Vector &el_dudy)
{
    // const int dof = el.GetDof();

    el_dudx.SetSize(dof * num_equations);
    el_dudy.SetSize(dof * num_equations);

    el_dudx = 0.0;
    el_dudy = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    DenseMatrix el_dudx_mat(el_dudx.GetData(), dof, num_equations);
    DenseMatrix el_dudy_mat(el_dudy.GetData(), dof, num_equations);

    grad_mat1.GetColumnReference(0, el_dudxi);
    grad_mat1.GetColumnReference(1, el_dudeta);

    for (int j = 0; j < Np_y; j++)
    {
        for (int i = 0; i < Np_x; i++)
        {
            id = j * Np_x + i;
            const IntegrationPoint &ip1 = ir_vol->IntPoint(id);
            Tr.SetIntPoint(&ip1);
            J = Tr.Weight();
            adj = Tr.AdjugateJacobian();

            grad_mat1 = 0.0;

            for (int l = 0; l < Np_x; l++)
            {
                el_u_mat.GetRow(j * Np_x + l, dU);
                dU *= D_T(l, i);
                el_dudxi += dU;

                el_u_mat.GetRow(l * Np_x + i, dU);
                dU *= D_T(l, j);
                el_dudeta += dU;
            }

            mfem::Mult(grad_mat1, adj, grad_mat2);
            grad_mat2 *= 1.0 / J;

            grad_mat2.GetColumn(0, grad_state);
            el_dudx_mat.SetRow(id, grad_state);

            grad_mat2.GetColumn(1, grad_state);
            el_dudy_mat.SetRow(id, grad_state);
        }
    }
}

void LiftingBR1::AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx)
{
    // const int dof = el.GetDof();

    el_dudx.SetSize(dof * num_equations);

    el_dudx = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    DenseMatrix el_dudx_mat(el_dudx.GetData(), dof, num_equations);

    grad_mat1.GetColumnReference(0, el_dudxi);

    for (int i = 0; i < Np_x; i++)
    {
        const IntegrationPoint &ip1 = ir_vol->IntPoint(i);
        Tr.SetIntPoint(&ip1);
        J = Tr.Weight();
        adj = Tr.AdjugateJacobian();

        grad_mat1 = 0.0;

        for (int l = 0; l < Np_x; l++)
        {
            el_u_mat.GetRow(l, dU);
            dU *= D_T(l, i);
            el_dudxi += dU;
        }

        mfem::Mult(grad_mat1, adj, grad_mat2);
        grad_mat2 *= 1.0 / J;

        grad_mat2.GetColumn(0, grad_state);
        el_dudx_mat.SetRow(i, grad_state);
    }
}

void LiftingBR1::AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables,
                                              Vector &el_dudx, Vector &el_dudy, Vector &el_dudz)
{ 
    el_dudx.SetSize(dof1 * num_equations);
    el_dudy.SetSize(dof1 * num_equations);
    el_dudz.SetSize(dof1 * num_equations);
    el_dudx = 0.0;
    el_dudy = 0.0;
    el_dudz = 0.0;
 
    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
    DenseMatrix el_dudz_mat1(el_dudz.GetData(), dof1, num_equations);
 
    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir_face->IntPoint(i);
       Tr.SetAllIntPoints(&ip);
       J1 = Tr.GetElement1Transformation().Weight();
       el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
 
       el_u_mat1.MultTranspose(shape1, state1);
 
       CalcOrtho(Tr.Jacobian(), nor);

       bfi->ComputeBdrFaceLiftingFlux(state1, f, Tr, ip, thermoTables);
       h = g = f;

       f *= nor(0);
       AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, f, el_dudx_mat1);
 
       g *= nor(1);
       AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, g, el_dudy_mat1);
 
       h *= nor(2);
       AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, h, el_dudz_mat1);       
    }
}

void LiftingBR1::AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables,
                                              Vector &el_dudx, Vector &el_dudy)
{ 
    el_dudx.SetSize(dof1 * num_equations);
    el_dudy.SetSize(dof1 * num_equations);
    el_dudx = 0.0;
    el_dudy = 0.0;
 
    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
 
    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir_face->IntPoint(i);
       Tr.SetAllIntPoints(&ip);
       J1 = Tr.GetElement1Transformation().Weight();
       el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
 
       el_u_mat1.MultTranspose(shape1, state1);
 
       CalcOrtho(Tr.Jacobian(), nor);

       bfi->ComputeBdrFaceLiftingFlux(state1, f, Tr, ip, thermoTables);
       g = f;

       f *= nor(0);
       AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, f, el_dudx_mat1);
 
       g *= nor(1);
       AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, g, el_dudy_mat1);      
    }
}

void LiftingBR1::AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables,
                                              Vector &el_dudx)
{ 
    el_dudx.SetSize(dof1 * num_equations);
    el_dudx = 0.0;
 
    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
 
    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir_face->IntPoint(i);
       Tr.SetAllIntPoints(&ip);
       J1 = Tr.GetElement1Transformation().Weight();
       el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
 
       el_u_mat1.MultTranspose(shape1, state1);
 
       nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;

       bfi->ComputeBdrFaceLiftingFlux(state1, f, Tr, ip, thermoTables);

       f *= nor(0);
       AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, f, el_dudx_mat1);      
    }
}


}
