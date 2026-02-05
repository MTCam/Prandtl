#include "DGSEMNonlinearForm.hpp"

namespace Prandtl
{

DGSEMNonlinearForm::DGSEMNonlinearForm(ParFiniteElementSpace *pf)
    : ParNonlinearForm(pf)
{
    GRAD_X.MakeRef(pf, NULL);
    GRAD_Y.MakeRef(pf, NULL);
    GRAD_Z.MakeRef(pf, NULL);
}

void DGSEMNonlinearForm::MultLifting(const Vector &u, Vector &dudx) const
{
    const Vector &pu = Prolongate(u);
    if (P)
    {
        aux2_x.SetSize(P->Height());
    }

    Vector &pdudx = P ? aux2_x : dudx;

    Array<int> vdofs;
    Vector el_u, el_dudx;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudx = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleLiftingElementVector(*fe, *T, el_u, el_dudx);
                pdudx.AddElementVector(vdofs, el_dudx);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append (vdofs2);

                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx);
                    pdudx.AddElementVector(vdofs, el_dudx);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();
            X.MakeRef(aux1, 0); // aux1 contains P.x
            X.ExchangeFaceNbrData();
            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());
                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx);
                    aux2_x.AddElementVector(vdofs1, el_dudx.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes -> GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations (i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx);
                    pdudx.AddElementVector(vdofs, el_dudx);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudx, dudx);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudx(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2_x, dudx);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto GRADU_X_RW = dudx.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { GRADU_X_RW[idx[i]] = 0.0; });
    }
}

void DGSEMNonlinearForm::MultLifting(const Vector &u, Vector &dudx, Vector &dudy) const
{
    const Vector &pu = Prolongate(u);
    if (P)
    {
        aux2_x.SetSize(P->Height());
        aux2_y.SetSize(P->Height());
    }

    Vector &pdudx = P ? aux2_x : dudx;
    Vector &pdudy = P ? aux2_y : dudy;

    Array<int> vdofs;
    Vector el_u, el_dudx, el_dudy;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudx = 0.0;
    pdudy = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleLiftingElementVector(*fe, *T, el_u, el_dudx, el_dudy);
                pdudx.AddElementVector(vdofs, el_dudx);
                pdudy.AddElementVector(vdofs, el_dudy);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append(vdofs2);

                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy);
                    pdudx.AddElementVector(vdofs, el_dudx);
                    pdudy.AddElementVector(vdofs, el_dudy);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();
            X.MakeRef(aux1, 0); // aux1 contains P.x
            X.ExchangeFaceNbrData();
            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());
                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy);
                    aux2_x.AddElementVector(vdofs1, el_dudx.GetData());
                    aux2_y.AddElementVector(vdofs1, el_dudy.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes -> GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations (i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy);
                    pdudx.AddElementVector(vdofs, el_dudx);
                    pdudy.AddElementVector(vdofs, el_dudy);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudx, dudx);
            cP->MultTranspose(pdudy, dudy);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudx(ess_tdof_list[i]) = 0.0;
            dudy(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2_x, dudx);
        P->MultTranspose(aux2_y, dudy);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto GRADU_X_RW = dudx.ReadWrite();
        auto GRADU_Y_RW = dudy.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { GRADU_X_RW[idx[i]] = 0.0; });
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { GRADU_Y_RW[idx[i]] = 0.0; });
    }
}

void DGSEMNonlinearForm::AssembleDeviceCache(Prandtl::DGSEMCache &dgsem_device_cache)
  {
    MFEM_VERIFY(fes, "fes must be set");
    Mesh *mesh = fes->GetMesh();
    MFEM_VERIFY(mesh, "mesh must be set");
    
    // ---- sizes / metadata ----------------------------------------------------
    const int ne = fes->GetNE();
    dgsem_device_cache.num_el = ne;
    // Attribute count = max attribute id (1-based in MFEM)
    dgsem_device_cache.num_attr = mesh->attributes.Size() ? mesh->attributes.Max() : 0;
    
    // ---- 1) Build combined attribute marker exactly like Mult() --------------
    dgsem_device_cache.attr_marker.SetSize(dgsem_device_cache.num_attr);
    dgsem_device_cache.attr_marker = 0;
    
    if (dnfi.Size() == 0 || dgsem_device_cache.num_attr == 0)
      {
        // If no domain integrators, nothing to do; marker stays 0.
        // If "process all" is desired instead, set marker=1 here.
      }
    else
      {
        for (int k = 0; k < dnfi.Size(); k++)
          {
            if (dnfi_marker[k] == nullptr)
              {
                dgsem_device_cache.attr_marker = 1; // process all attrs
                break;
              }

            const Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == dgsem_device_cache.attr_marker.Size(),
                        "invalid marker for domain integrator #" << k);
            
            for (int i = 0; i < dgsem_device_cache.attr_marker.Size(); i++)
              {
                dgsem_device_cache.attr_marker[i] |= marker[i];
              }
          }
      }

    // ---- 2) Per-element attribute id array -----------------------------------
    dgsem_device_cache.elem_attr.SetSize(ne);
    for (int e = 0; e < ne; ++e)
      {
        const int attr = mesh->GetAttribute(e); // 1-based
        dgsem_device_cache.elem_attr[e] = attr;
      }

    // Optional host-side sanity check (cheap, catches bad markers early):
    if (dgsem_device_cache.num_attr > 0)
      {
        for (int e = 0; e < ne; ++e)
          {
            const int a = dgsem_device_cache.elem_attr[e];
            MFEM_VERIFY(a >= 1 && a <= dgsem_device_cache.num_attr,
                        "element attribute out of range: attr=" << a
                        << " num_attr=" << dgsem_device_cache.num_attr);
          }
      }
  }
  
void DGSEMNonlinearForm::MultLifting(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const
{
    const Vector &pu = Prolongate(u);
    if (P)
    {
        aux2_x.SetSize(P->Height());
        aux2_y.SetSize(P->Height());
        aux2_z.SetSize(P->Height());
    }

    Vector &pdudx = P ? aux2_x : dudx;
    Vector &pdudy = P ? aux2_y : dudy;
    Vector &pdudz = P ? aux2_z : dudz;

    Array<int> vdofs;
    Vector el_u, el_dudx, el_dudy, el_dudz;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudx = 0.0;
    pdudy = 0.0;
    pdudz = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleLiftingElementVector(*fe, *T, el_u, el_dudx, el_dudy, el_dudz);
                pdudx.AddElementVector(vdofs, el_dudx);
                pdudy.AddElementVector(vdofs, el_dudy);
                pdudz.AddElementVector(vdofs, el_dudz);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append (vdofs2);

                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudz);
                    pdudx.AddElementVector(vdofs, el_dudx);
                    pdudy.AddElementVector(vdofs, el_dudy);
                    pdudz.AddElementVector(vdofs, el_dudz);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();
            X.MakeRef(aux1, 0); // aux1 contains P.x
            X.ExchangeFaceNbrData();
            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());
                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudz);
                    aux2_x.AddElementVector(vdofs1, el_dudx.GetData());
                    aux2_y.AddElementVector(vdofs1, el_dudy.GetData());
                    aux2_z.AddElementVector(vdofs1, el_dudz.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes -> GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations (i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleLiftingFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudz);
                    pdudx.AddElementVector(vdofs, el_dudx);
                    pdudy.AddElementVector(vdofs, el_dudy);
                    pdudz.AddElementVector(vdofs, el_dudz);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudx, dudx);
            cP->MultTranspose(pdudy, dudy);
            cP->MultTranspose(pdudz, dudz);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudx(ess_tdof_list[i]) = 0.0;
            dudy(ess_tdof_list[i]) = 0.0;
            dudz(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2_x, dudx);
        P->MultTranspose(aux2_y, dudy);
        P->MultTranspose(aux2_z, dudz);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto GRADU_X_RW = dudx.ReadWrite();
        auto GRADU_Y_RW = dudy.ReadWrite();
        auto GRADU_Z_RW = dudz.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { GRADU_X_RW[idx[i]] = 0.0; });
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { GRADU_Y_RW[idx[i]] = 0.0; });
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { GRADU_Z_RW[idx[i]] = 0.0; });
    }
}

void DGSEMNonlinearForm::Mult(const Vector &u, Vector &dudt) const
{
    const Vector &pu = Prolongate(u);
    
    if (P)
    {
        aux2.SetSize(P->Height());
    }
    Vector &pdudt = P ? aux2 : dudt;

    Array<int> vdofs;
    Vector el_u, el_dudt;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudt = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleElementVector(*fe, *T, el_u, el_dudt);
                pdudt.AddElementVector(vdofs, el_dudt);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append (vdofs2);

                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();

            X.MakeRef(aux1, 0); // aux1 contains P.x

            X.ExchangeFaceNbrData();

            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());

                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudt);
                    aux2.AddElementVector(vdofs1, el_dudt.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes->GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);

                pu.GetSubVector(vdofs, el_u);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudt, dudt);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudt(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2, dudt);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto DU_RW = dudt.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });
    }
}

void DGSEMNonlinearForm::Mult(const Vector &u, const Vector &dudx, Vector &dudt) const
{
    const Vector &pu = Prolongate(u);
    
    if (P)
    {
        aux2.SetSize(P->Height());

        aux2_x.SetSize(P->Height());

        P->Mult(dudx, aux2_x);
    }
    Vector &pdudt = P ? aux2 : dudt;
    const Vector &pdudx = P ? aux2_x : dudx;

    Array<int> vdofs;
    Vector el_u, el_dudt, el_dudx;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudt = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            pdudx.GetSubVector(vdofs, el_dudx);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleElementVector(*fe, *T, el_u, el_dudx, el_dudt);
                pdudt.AddElementVector(vdofs, el_dudt);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append (vdofs2);

                pu.GetSubVector(vdofs, el_u);
                pdudx.GetSubVector(vdofs, el_dudx);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();
            aux2_x.HostReadWrite();

            X.MakeRef(aux1, 0); // aux1 contains P.x
            GRAD_X.MakeRef(aux2_x, 0);

            X.ExchangeFaceNbrData();
            GRAD_X.ExchangeFaceNbrData();

            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());
                el_dudx.SetSize(vdofs1.Size() + vdofs2.Size());

                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                GRAD_X.GetSubVector(vdofs1, el_dudx.GetData());
                GRAD_X.FaceNbrData().GetSubVector(vdofs2, el_dudx.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudt);
                    aux2.AddElementVector(vdofs1, el_dudt.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes->GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);

                pu.GetSubVector(vdofs, el_u);
                pdudx.GetSubVector(vdofs, el_dudx);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudt, dudt);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudt(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2, dudt);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto DU_RW = dudt.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });
    }
}

void DGSEMNonlinearForm::Mult(const Vector &u, const Vector &dudx, const Vector &dudy, Vector &dudt) const
{
    const Vector &pu = Prolongate(u);
    
    if (P)
    {
        aux2.SetSize(P->Height());

        aux2_x.SetSize(P->Height());
        aux2_y.SetSize(P->Height());

        P->Mult(dudx, aux2_x);
        P->Mult(dudy, aux2_y);
    }
    Vector &pdudt = P ? aux2 : dudt;
    const Vector &pdudx = P ? aux2_x : dudx;
    const Vector &pdudy = P ? aux2_y : dudy;

    Array<int> vdofs;
    Vector el_u, el_dudt, el_dudx, el_dudy;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudt = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            pdudx.GetSubVector(vdofs, el_dudx);
            pdudy.GetSubVector(vdofs, el_dudy);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleElementVector(*fe, *T, el_u, el_dudx, el_dudy, el_dudt);
                pdudt.AddElementVector(vdofs, el_dudt);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append(vdofs2);

                pu.GetSubVector(vdofs, el_u);
                pdudx.GetSubVector(vdofs, el_dudx);
                pdudy.GetSubVector(vdofs, el_dudy);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();
            aux2_x.HostReadWrite();
            aux2_y.HostReadWrite();

            X.MakeRef(aux1, 0); // aux1 contains P.x
            GRAD_X.MakeRef(aux2_x, 0);
            GRAD_Y.MakeRef(aux2_y, 0);

            X.ExchangeFaceNbrData();
            GRAD_X.ExchangeFaceNbrData();
            GRAD_Y.ExchangeFaceNbrData();

            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());
                el_dudx.SetSize(vdofs1.Size() + vdofs2.Size());
                el_dudy.SetSize(vdofs1.Size() + vdofs2.Size());

                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                GRAD_X.GetSubVector(vdofs1, el_dudx.GetData());
                GRAD_X.FaceNbrData().GetSubVector(vdofs2, el_dudx.GetData() + vdofs1.Size());

                GRAD_Y.GetSubVector(vdofs1, el_dudy.GetData());
                GRAD_Y.FaceNbrData().GetSubVector(vdofs2, el_dudy.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudt);
                    aux2.AddElementVector(vdofs1, el_dudt.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes->GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);

                pu.GetSubVector(vdofs, el_u);
                pdudx.GetSubVector(vdofs, el_dudx);
                pdudy.GetSubVector(vdofs, el_dudy);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudt, dudt);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudt(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2, dudt);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto DU_RW = dudt.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });
    }
}

void DGSEMNonlinearForm::Mult(const Vector &u, const Vector &dudx, const Vector &dudy, const Vector &dudz, Vector &dudt) const
{
    const Vector &pu = Prolongate(u);
    
    if (P)
    {
        aux2.SetSize(P->Height());

        aux2_x.SetSize(P->Height());
        aux2_y.SetSize(P->Height());
        aux2_z.SetSize(P->Height());

        P->Mult(dudx, aux2_x);
        P->Mult(dudy, aux2_y);
        P->Mult(dudz, aux2_z);
    }
    Vector &pdudt = P ? aux2 : dudt;
    const Vector &pdudx = P ? aux2_x : dudx;
    const Vector &pdudy = P ? aux2_y : dudy;
    const Vector &pdudz = P ? aux2_z : dudz;

    Array<int> vdofs;
    Vector el_u, el_dudt, el_dudx, el_dudy, el_dudz;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    pdudt = 0.0;

    if (dnfi.Size())
    {
        // Which attributes need to be processed?
        Array<int> attr_marker(mesh->attributes.Size() ?
                                mesh->attributes.Max() : 0);
        attr_marker = 0;
        for (int k = 0; k < dnfi.Size(); k++)
        {
            if (dnfi_marker[k] == NULL)
            {
                attr_marker = 1;
                break;
            }
            Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == attr_marker.Size(),
                        "invalid marker for domain integrator #"
                        << k << ", counting from zero");
            for (int i = 0; i < attr_marker.Size(); i++)
            {
                attr_marker[i] |= marker[i];
            }
        }
 
        for (int i = 0; i < fes->GetNE(); i++)
        {
            const int attr = mesh->GetAttribute(i);
            if (attr_marker[attr-1] == 0) { continue; }

            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            pdudx.GetSubVector(vdofs, el_dudx);
            pdudy.GetSubVector(vdofs, el_dudy);
            pdudz.GetSubVector(vdofs, el_dudz);
            for (int k = 0; k < dnfi.Size(); k++)
            {
                if (dnfi_marker[k] &&
                    (*dnfi_marker[k])[attr-1] == 0) { continue; }

                dnfi[k]->AssembleElementVector(*fe, *T, el_u, el_dudx, el_dudy, el_dudz, el_dudt);
                pdudt.AddElementVector(vdofs, el_dudt);
            }
        }
    }

    if (fnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;
        Array<int> vdofs2;

        for (int i = 0; i < mesh->GetNumFaces(); i++)
        {
            tr = mesh->GetInteriorFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);
                fes->GetElementVDofs(tr->Elem2No, vdofs2);
                vdofs.Append (vdofs2);

                pu.GetSubVector(vdofs, el_u);
                pdudx.GetSubVector(vdofs, el_dudx);
                pdudy.GetSubVector(vdofs, el_dudy);
                pdudz.GetSubVector(vdofs, el_dudz);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fes->GetFE(tr->Elem2No);

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudz, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
        if (!Serial())
        {
            // Terms over shared interior faces in parallel.
            ParFiniteElementSpace *pfes = ParFESpace();
            ParMesh *pmesh = pfes->GetParMesh();
            FaceElementTransformations *tr;
            const FiniteElement *fe1, *fe2;
            Array<int> vdofs1, vdofs2;

            aux1.HostReadWrite();
            aux2_x.HostReadWrite();
            aux2_y.HostReadWrite();
            aux2_z.HostReadWrite();

            X.MakeRef(aux1, 0); // aux1 contains P.x
            GRAD_X.MakeRef(aux2_x, 0);
            GRAD_Y.MakeRef(aux2_y, 0);
            GRAD_Z.MakeRef(aux2_z, 0);

            X.ExchangeFaceNbrData();
            GRAD_X.ExchangeFaceNbrData();
            GRAD_Y.ExchangeFaceNbrData();
            GRAD_Z.ExchangeFaceNbrData();

            const int n_shared_faces = pmesh->GetNSharedFaces();
            for (int i = 0; i < n_shared_faces; i++)
            {
                tr = pmesh->GetSharedFaceTransformations(i, true);
                int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

                fe1 = pfes->GetFE(tr->Elem1No);
                fe2 = pfes->GetFaceNbrFE(Elem2NbrNo);

                pfes->GetElementVDofs(tr->Elem1No, vdofs1);
                pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);

                el_u.SetSize(vdofs1.Size() + vdofs2.Size());
                el_dudx.SetSize(vdofs1.Size() + vdofs2.Size());
                el_dudy.SetSize(vdofs1.Size() + vdofs2.Size());
                el_dudz.SetSize(vdofs1.Size() + vdofs2.Size());

                X.GetSubVector(vdofs1, el_u.GetData());
                X.FaceNbrData().GetSubVector(vdofs2, el_u.GetData() + vdofs1.Size());

                GRAD_X.GetSubVector(vdofs1, el_dudx.GetData());
                GRAD_X.FaceNbrData().GetSubVector(vdofs2, el_dudx.GetData() + vdofs1.Size());

                GRAD_Y.GetSubVector(vdofs1, el_dudy.GetData());
                GRAD_Y.FaceNbrData().GetSubVector(vdofs2, el_dudy.GetData() + vdofs1.Size());

                GRAD_Z.GetSubVector(vdofs1, el_dudz.GetData());
                GRAD_Z.FaceNbrData().GetSubVector(vdofs2, el_dudz.GetData() + vdofs1.Size());

                for (int k = 0; k < fnfi.Size(); k++)
                {
                    fnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudz, el_dudt);
                    aux2.AddElementVector(vdofs1, el_dudt.GetData());
                }
            }
        }
    }

    if (bfnfi.Size())
    {
        FaceElementTransformations *tr;
        const FiniteElement *fe1, *fe2;

        // Which boundary attributes need to be processed?
        Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                    mesh->bdr_attributes.Max() : 0);
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

        for (int i = 0; i < fes->GetNBE(); i++)
        {
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

            tr = mesh->GetBdrFaceTransformations(i);
            if (tr != NULL)
            {
                fes->GetElementVDofs(tr->Elem1No, vdofs);

                pu.GetSubVector(vdofs, el_u);
                pdudx.GetSubVector(vdofs, el_dudx);
                pdudy.GetSubVector(vdofs, el_dudy);
                pdudz.GetSubVector(vdofs, el_dudz);

                fe1 = fes->GetFE(tr->Elem1No);
                fe2 = fe1;
                for (int k = 0; k < bfnfi.Size(); k++)
                {
                    if (bfnfi_marker[k] &&
                        (*bfnfi_marker[k])[bdr_attr-1] == 0) { continue; }

                    bfnfi[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_u, el_dudx, el_dudy, el_dudz, el_dudt);
                    pdudt.AddElementVector(vdofs, el_dudt);
                }
            }
        }
    }

    if (Serial())
    {
        if (cP)
        {
            cP->MultTranspose(pdudt, dudt);
        }

        for (int i = 0; i < ess_tdof_list.Size(); i++)
        {
            dudt(ess_tdof_list[i]) = 0.0;
        }
    }
    else
    {
        P->MultTranspose(aux2, dudt);

        const int N = ess_tdof_list.Size();
        const auto idx = ess_tdof_list.Read();
        auto DU_RW = dudt.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });
    }
}

}

