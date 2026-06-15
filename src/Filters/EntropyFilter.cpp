#include "EntropyFilter.hpp"
#include "BasicOperations.hpp"
#include "BdrFaceIntegrator.hpp"

namespace Prandtl
{

EntropyFilter::EntropyFilter(
    std::shared_ptr<ParFiniteElementSpace> vfes_,
    std::shared_ptr<ParFiniteElementSpace> fes0_,
    std::shared_ptr<ParGridFunction> sol_,
    std::shared_ptr<ParMesh> mesh_,
    std::unique_ptr<ParGridFunction> entropy_,
    std::unique_ptr<ModalBasis> modalBasis_,
    const Table &el2el, const Table &el2bdrel,
    const IntegrationRule *vol_ir_, const IntegrationRule *face_ir_,
    const IntegrationRule *bdr_face_ir_, real_t gamma
    ) : Filter(), vfes(vfes_), fes0(fes0_), sol(sol_), mesh(mesh_),
        entropy(std::move(entropy_)), modalBasis(std::move(modalBasis_)),
        element2element(el2el), element2bdrelement(el2bdrel),
        vol_ir(vol_ir_), face_ir(face_ir_), bdr_face_ir(bdr_face_ir_),
        VDM(modalBasis->GetVandermonde()), VDM_vol_ir(modalBasis->ComputeVDM(vol_ir_)),
        dim(mesh->SpaceDimension()), ndofs(vfes->GetFE(0)->GetDof()),
        num_equations(vfes->GetVDim()), num_elements(mesh->GetNE()),
        order(vfes->GetElementOrder(0)), element(mesh->GetElement(0)),
        etype(element->GetType()), egeom(element->GetGeometryType()),
        ftype(dim == 2 ? Element::SEGMENT : (element->GetNFaceVertices(0) == 3 ? Element::TRIANGLE : Element::QUADRILATERAL)),
        fgeom(dim == 2 ? Geometry::SEGMENT : (element->GetNFaceVertices(0) == 3 ? Geometry::TRIANGLE : Geometry::SQUARE)),
        gamma(gamma), gammaM1(gamma - 1.0)
{
    max_order.SetSize(ndofs);
    VDM_face_irs.resize(Geometry::NumBdrArray[egeom]);
    mapped_face_ir.SetSize(face_ir->GetNPoints());
    for (int f = 0; f < Geometry::NumBdrArray[egeom]; f++)
    {
        VDM_face_irs[f].SetSize(face_ir->GetNPoints(), ndofs);
        mesh->GetLocalFaceTransformation(ftype, etype, Tr_ip.Transf, 64 * f);
        Tr_ip.Transform(*face_ir, mapped_face_ir);
        VDM_face_irs[f] = modalBasis->ComputeVDM(&mapped_face_ir);
    }

    ffac.SetSize(order + 1);
    ffac = 1.0;
    nor.SetSize(dim); 

    shape.SetSize(ndofs);
    vdof_col.SetSize(ndofs);
    modes_col.SetSize(ndofs);

    state1.SetSize(num_equations);
    state2.SetSize(num_equations);

    modes_mat.SetSize(ndofs, num_equations);
    vdof_mat1.SetSize(ndofs, num_equations);
    vdof_mat2.SetSize(ndofs, num_equations);

    hierarch_states.SetSize(order + 1, num_equations);

    for (int i = 0; i < ndofs; i++)
    {
        if (dim == 1)
        {
            max_order[i] = modalBasis->GetPolyDegs()(i, 0);
        }
        else
        {
            max_order[i] = std::max(modalBasis->GetPolyDegs()(i, 0), modalBasis->GetPolyDegs()(i, 1));
            if (dim == 3)
            {
                max_order[i] = std::max(max_order[i], modalBasis->GetPolyDegs()(i, 2));
            }
        }
    }
}

void EntropyFilter::GetFilterConstraints(const Vector &x)
{
    GetEntropyConstraints(x);
}

void EntropyFilter::GetEntropyConstraints(const Vector &x)
{
    *sol = x;
    sol->ExchangeFaceNbrData();

    if (bfnfi.Size())
    {
        // bdr_attr_marker.SetSize(mesh->bdr_attributes.Size() ?
        //                             mesh->bdr_attributes.Max() : 0);
        bdr_attr_marker.SetSize(mesh->bdr_attributes.Size());
        bdr_attr_marker = 0;
        for (int k = 0; k < bfnfi.Size(); k++)
        {
            if (bfnfi_marker[k] == NULL)
            {
                bdr_attr_marker = 1;
                break;
            }
            Array<int> &bdr_marker = *bfnfi_marker[k];
            MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                        "invalid boundary marker for boundary face integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < bdr_attr_marker.Size(); i++)
            {
                bdr_attr_marker[i] |= bdr_marker[i];
            }
        }
    }

    Array<int> el2el, el2bdrel, vdof_indices, ent_indices;
    int indx, shifted_indx;

    for (int el = 0; el < vfes->GetNE(); el++)
    {
        element2element.GetRow(el, el2el);
        element2bdrelement.GetRow(el, el2bdrel);
        el2el.Prepend(el);
        fes0->GetElementVDofs(el, ent_indices);

        el_min_e = __DBL_MAX__;
        for (int i = 0; i < el2el.Size(); i++)
        {
            indx = el2el[i];
            fe2 = vfes->GetFE(indx);
            MFEM_ASSERT(fe2 != nullptr, "fe2 is not initialized");

            if (indx >= num_elements)
            {
                shifted_indx = indx - num_elements;
                vfes->GetFaceNbrElementVDofs(shifted_indx, vdof_indices);
                sol->FaceNbrData().GetSubVector(vdof_indices, el_vdofs);
            }
            else
            {
                vfes->GetElementVDofs(indx, vdof_indices);
                sol->GetSubVector(vdof_indices, el_vdofs);
                // el_vdofs.Print(std::cout);
            }

            vdof_mat2.UseExternalData(el_vdofs.GetData(), ndofs, num_equations);

            if (i == 0)
            {
                fe1 = fe2;
                vdof_mat1 = vdof_mat2;
                MFEM_ASSERT(fe1 != nullptr, "fe1 is not initialized");
            }

            ComputeElementMinima(vdof_mat2, fe2, state2);
        }

        ComputeBdrStateMinima(vdof_mat1, fe1, el2bdrel, state1, state2);
        entropy->SetSubVector(ent_indices, el_min_e);

        // vdof_mat1.ClearExternalData();
        vdof_mat2.ClearExternalData();
    }
}

void EntropyFilter::FilterModes(const DenseMatrix &hiearch_states, Vector &state, real_t zeta)
{
    hierarch_states.GetRow(0, state);
    for (int i = 1; i < hierarch_states.Height(); i++)
    {
        hierarch_states.GetRow(i, state2);
        state2 *= exp(-zeta * pow(i, 2));
        state += state2;
    }
}

void EntropyFilter::FilterSolution(Vector &x)
{
    Array<int> vdof_indices, ent_indices;

    for (int el = 0; el < vfes->GetNE(); el++)
    {
        vfes->GetElementVDofs(el, vdof_indices);
        fes0->GetElementVDofs(el, ent_indices);
        x.GetSubVector(vdof_indices, el_vdofs);
        entropy->GetSubVector(ent_indices, ent_dof);
        vdof_mat1.UseExternalData(el_vdofs.GetData(), ndofs, num_equations);

        el_min_rho = el_min_p = el_min_e = infinity();

        for (int i = 0; i < ndofs; i++)
        {
            vdof_mat1.GetRow(i, state1);
            ComputeMinima(state1, "all");
        }

        if (el_min_rho < rho_min || el_min_p < p_min || el_min_e < ent_dof(0) - ent_tol)
        {
            for (int eq = 0; eq < num_equations; eq++)
            {
                vdof_mat1.GetColumn(eq, vdof_col);
                modes_mat.GetColumnReference(eq, modes_col);
                modalBasis->ComputeModes(vdof_col, modes_col);
            }

            zeta = 0.0;
            for (int i = 0; i < ndofs; i++)
            {
                ComputeHierarchStates(VDM, hierarch_states, i);
                FilterModes(hierarch_states, state1, zeta);
                rho = state1(0);
                p = ComputePressure(state1, gammaM1);
                e = ComputeEntropy(rho, p, gamma);
                if (rho < rho_min || p < p_min || e < ent_dof(0) - ent_tol)
                {
                    ComputeRoot(state1, zeta);
                }
            }
            
            for (int d = 1; d < order + 1; ++d)
            {
                ffac(d) = std::exp(-zeta * pow(d, 2));
            }

            for (int i = 0; i < ndofs; i++)
            {
                for (int k = 0; k < num_equations; k++)
                {
                    tmp = 0.0;
                    for (int j = 0; j < order + 1; j++)
                    {
                        for (int l = 0; l < ndofs; l++)
                        {
                            if (max_order[l] == j)
                            {
                                tmp += ffac(j) * VDM(i, l) * modes_mat(l, k);
                                // tmp += exp(-zeta * pow(max_order[l], 2)) * VDM(i, l) * modes_mat(l, k);
                            }
                        }
                    }
                    vdof_mat1(i, k) = tmp;
                }
            }
            x.SetSubVector(vdof_indices, el_vdofs);
        }
        vdof_mat1.ClearExternalData();
    }
}

void EntropyFilter::ComputeHierarchStates(const DenseMatrix &VDM, DenseMatrix &hierarch_states, int row)
{
    hierarch_states = 0.0;
    for (int j = 0; j < order + 1; j++)
    {
        for (int k = 0; k < num_equations; k++)
        {
            for (int l = 0; l < ndofs; l++)
            {
                if (max_order[l] == j)
                {
                    hierarch_states(j, k) += VDM(row, l) * modes_mat(l, k);
                }
            }
        }
    }
}

void EntropyFilter::ComputeMinima(const Vector &state, const char *mode)
{
    rho = state(0);
    p = ComputePressure(state, gammaM1);
    e = ComputeEntropy(rho, p, gamma);
    el_min_e = std::min(el_min_e, e);
    if (strcmp(mode, "all") == 0)
    {
        el_min_rho = std::min(el_min_rho, rho);
        el_min_p = std::min(el_min_p, p);
    }
}

void EntropyFilter::ComputeElementMinima(const DenseMatrix &vdof_mat, const FiniteElement *fe, Vector &state, const char *mode)
{
}

void EntropyFilter::ComputeBdrStateMinima(const DenseMatrix &vdof_mat,
    const FiniteElement *fe, const Array<int> &el2bdrel, Vector &state1, Vector &state2, const char *mode)
{
    if (bfnfi.Size())
    {
        for (int m = 0; m < el2bdrel.Size(); m++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(el2bdrel[m]);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            Tr_face = mesh->GetBdrFaceTransformations(el2bdrel[m]);

            if (Tr_face != NULL)
            {
                for (int n = 0; n < bfnfi.Size(); n++)
                {
                    if (bfnfi_marker[n] && (*bfnfi_marker[n])[bdr_attr-1] == 0)
                    {
                        continue;
                    }
                    for (int i = 0; i < bdr_face_ir->GetNPoints(); i++)
                    {
                        const IntegrationPoint &ip = bdr_face_ir->IntPoint(i);
                        Tr_face->SetAllIntPoints(&ip);
                        fe->CalcShape(Tr_face->GetElement1IntPoint(), shape);
                        vdof_mat.MultTranspose(shape, state1);

                        if (dim == 1) 
                        {
                            nor(0) = (Tr_face->GetElement1IntPoint().x - 0.5) * 2.0;
                        }
                        else
                        {
                            CalcOrtho(Tr_face->Jacobian(), nor);
                        } 

                        state2 = 0.0;
                        ThermoTablesView thermoTables; // CL ALERT : Just a placeholder to avail the function call
#ifdef LTE_EOS
    MFEM_ABORT("CL ALERT : Entropy Filer is not functional with LTE_EOS");
#endif
                        bfnfi[n]->ComputeOuterInviscidState(state1, state2, *Tr_face, ip, thermoTables);
                        ComputeMinima(state2, mode);
                    }
                }
            }
        }
    }
}

void EntropyFilter::ComputeRoot(Vector &state, real_t &zeta)
{
    zeta1 = zeta;
    zeta2 = -std::log(epsilon);
    
    // for (int iter = 0; iter < N_iter && (zeta2 - zeta1 > zeta_tol || p < p_min || rho < rho_min || e < ent_dof(0) - ent_tol); iter++)
    for (int iter = 0; iter < N_iter; iter++)
    {
        zeta3 = 0.5 * (zeta1 + zeta2);
        FilterModes(hierarch_states, state, zeta3);
        rho = state(0);
        p = ComputePressure(state, gammaM1);
        e = ComputeEntropy(rho, p, gamma);
        if (rho < rho_min || p < p_min || e < (*entropy)(0) - ent_tol)
        {
            zeta1 = zeta3;
        }
        else
        {
            zeta2 = zeta3;
        }

    }
    zeta = zeta2;   
}

}