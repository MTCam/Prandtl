#include "DGSEMNonlinearForm.hpp"
#include "LiftingBR1.hpp"

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

void DGSEMNonlinearForm::GradOperatorVolume(const Vector &pu, std::vector<mfem::Vector *> &p_grad_u) const
{

  const int dim = cache->dim;

  if (dim==2){

    mfem::Vector &pdudx(*p_grad_u[0]);
    mfem::Vector &pdudy(*p_grad_u[1]);

    mfem::Vector Ue(cache->restr_v->Height());
    mfem::Vector dUxe(cache->restr_v->Height());
    mfem::Vector dUye(cache->restr_v->Height());
    
    Ue.UseDevice();
    dUxe.UseDevice();
    dUye.UseDevice();
    
    cache->restr_v->Mult(pu, Ue);
    
    {
      real_t *dx = dUxe.Write();
      real_t *dy = dUye.Write();
      real_t *pdx = p_grad_u[0]->Write();
      real_t *pdy = p_grad_u[1]->Write();
      mfem::forall(dUxe.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
        dx[i] = real_t(0);
        dy[i] = real_t(0);
        pdx[i] = real_t(0);
        pdy[i] = real_t(0);
      });
    }

    const real_t *Ue_d = Ue.Read();
    real_t *dUxe_d = dUxe.Write();
    real_t *dUye_d = dUye.Write();

    auto dc = device_cache;

    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int estride = ndof * neq;
    const int jac_stride = ndof;
    const int metric_stride = ndof * dc.dim * dc.dim;
    
    const real_t *elJac_d = dc.elJac_d;
    const real_t *elMetric_d = dc.elMetric_d;
    
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      const real_t *u_el = Ue_d + e * estride;
      real_t *dux_el = dUxe_d + e * estride;
      real_t *duy_el = dUye_d + e * estride;
      
      const real_t *jac_el = elJac_d + e * jac_stride;
      const real_t *metric_el = elMetric_d + e * metric_stride;
      
      LiftingBR1::AssembleGradElementVolumeKernel2D(dc, u_el, jac_el, metric_el,
                                                    dux_el, duy_el);
    });
    
    cache->restr_v->AddMultTranspose(dUxe, pdudx);
    cache->restr_v->AddMultTranspose(dUye, pdudy);
    
  } else {
    MFEM_ASSERT(false, "dim != 2 unsupported right now.");
  }
}
  
  void DGSEMNonlinearForm::GradOperatorBoundaryFaces(const mfem::Vector &pu,
                                                     std::vector<mfem::Vector *> &p_grad_u) const
  {
    const int dim = cache->dim;
    MFEM_ASSERT(dim == 2, "Only dim=2 supported right now.");
    
    mfem::Vector &pdudx(*p_grad_u[0]);
    mfem::Vector &pdudy(*p_grad_u[1]);
    
    auto dc = device_cache;
    
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int face_size = nfp * neq;
    const int nfaces_restr = cache->restr_b->Height() / face_size;
    const int norm_size = nfp * dim;
    const int npoints_bnd = nfaces_restr * nfp;
    
    mfem::Vector u_faces(cache->restr_b->Height());
    mfem::Vector rhs_faces_x(cache->restr_b->Height());
    mfem::Vector rhs_faces_y(cache->restr_b->Height());
    mfem::Vector faces_dudx(pdudx.Size());
    mfem::Vector faces_dudy(pdudy.Size());
    
    u_faces.UseDevice();
    rhs_faces_x.UseDevice();
    rhs_faces_y.UseDevice();
    faces_dudx.UseDevice();
    faces_dudy.UseDevice();
    
    {
      real_t *rx = rhs_faces_x.Write();
      real_t *ry = rhs_faces_y.Write();
      real_t *dx = faces_dudx.Write();
      real_t *dy = faces_dudy.Write();
      
      mfem::forall(rhs_faces_x.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
        rx[i] = real_t(0);
        ry[i] = real_t(0);
      });
      
      mfem::forall(faces_dudx.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
        dx[i] = real_t(0);
        dy[i] = real_t(0);
      });
    }
    
    cache->restr_b->Mult(pu, u_faces);
    
    const real_t *u_d = u_faces.Read();
    real_t *rhs_x_d = rhs_faces_x.Write();
    real_t *rhs_y_d = rhs_faces_y.Write();
    
    const real_t *nor_d = dc.bnd_nor_d;
    const real_t *wt_d = dc.bnd_wt_d;
    const int *bnd_marker_index_d = dc.bnd_marker_index_d;
    
    mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
    {
      const int f = p / nfp;
      const int fp = p % nfp;
      
      const int bnd_face_marker_index = bnd_marker_index_d[f];
      if (bnd_face_marker_index < 0)
        {
          return;
        }
      
      const int bc_index = bnd_face_marker_index; // same convention as inviscid device path for now
      if (bc_index < 0)
        {
          return;
        }
      
      const Prandtl::BCDescriptor &bc = dc.bc_descr_d[bc_index];
      if (bc.type == int(Prandtl::BCType::Invalid))
        {
          return;
        }
      
      const int face_offset = f * face_size;
      const int norm_offset = f * norm_size;
      const int w_offset = f * nfp;
      
      const real_t *u_face_d = u_d + face_offset;
      real_t *rhs_face_x_d = rhs_x_d + face_offset;
      real_t *rhs_face_y_d = rhs_y_d + face_offset;
      
      const real_t *nor_face_d = nor_d + norm_offset;
      const real_t *nor_point = nor_face_d + fp * dim;
      
      // Legacy one-sided boundary lifting uses +1/(w0*J1)
      const real_t scale = wt_d[w_offset + fp];

      LiftingBR1::AssembleGradBoundaryPointKernel2D(dc, bc,
                                                    u_face_d,
                                                    nor_point,
                                                    scale,
                                                    fp,
                                                    rhs_face_x_d,
                                                    rhs_face_y_d);
    });
    
    cache->restr_b->MultTranspose(rhs_faces_x, faces_dudx);
    cache->restr_b->MultTranspose(rhs_faces_y, faces_dudy);
    
    pdudx += faces_dudx;
    pdudy += faces_dudy;
  }
  
  // void DGSEMNonlinearForm::GradOperatorBoundaryFacesDevice(
  //     const mfem::Vector &pu,
  //     std::vector<mfem::Vector *> &p_grad_u) const
  // {
  //   MFEM_ASSERT(cache->dim == 2, "Only dim=2 supported right now.");
  //   return;
  // }
  
  
void DGSEMNonlinearForm::GradOperatorInteriorFaces(const mfem::Vector &pu,
                                                   std::vector<mfem::Vector *> &p_grad_u) const
{
  const int dim = cache->dim;
  MFEM_ASSERT(dim == 2, "Only dim=2 supported right now.");

  mfem::Vector &pdudx(*p_grad_u[0]);
  mfem::Vector &pdudy(*p_grad_u[1]);

  auto dc = device_cache;

  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int nfaces = cache->restr_f->Height() / (2 * nfp * neq);
  const int face_size = 2 * nfp * neq;
  const int norm_size = nfp * dim;

  mfem::Vector u_faces(cache->restr_f->Height());
  mfem::Vector rhs_faces_x(cache->restr_f->Height());
  mfem::Vector rhs_faces_y(cache->restr_f->Height());

  mfem::Vector faces_dudx(pdudx.Size());
  mfem::Vector faces_dudy(pdudy.Size());

  u_faces.UseDevice();
  rhs_faces_x.UseDevice();
  rhs_faces_y.UseDevice();
  faces_dudx.UseDevice();
  faces_dudy.UseDevice();

  faces_dudx = 0.0;
  faces_dudy = 0.0;

  {
    real_t *rx = rhs_faces_x.Write();
    real_t *ry = rhs_faces_y.Write();
    mfem::forall(rhs_faces_x.Size(), [=] MFEM_HOST_DEVICE (int i)
    {
      rx[i] = real_t(0);
      ry[i] = real_t(0);
    });
  }

  cache->restr_f->Mult(pu, u_faces);

  const real_t *u_d = u_faces.Read();
  real_t *rhs_x_d = rhs_faces_x.Write();
  real_t *rhs_y_d = rhs_faces_y.Write();

  const real_t *nor_d  = dc.nor_d;
  const real_t *wm_d   = dc.fw_minus_d;
  const real_t *wp_d   = dc.fw_plus_d;

  mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int f)
  {
    const int face_offset = f * face_size;
    const int norm_offset = f * norm_size;
    const int w_offset    = f * nfp;

    const real_t *u_face_d    = u_d + face_offset;
    const real_t *nor_face_d  = nor_d + norm_offset;
    const real_t *w_minus_d   = wm_d + w_offset;
    const real_t *w_plus_d    = wp_d + w_offset;

    real_t *rhs_face_x_d = rhs_x_d + face_offset;
    real_t *rhs_face_y_d = rhs_y_d + face_offset;

    LiftingBR1::AssembleGradInteriorFaceKernel2D(dc,
                                                 u_face_d,
                                                 nor_face_d,
                                                 w_minus_d,
                                                 w_plus_d,
                                                 rhs_face_x_d,
                                                 rhs_face_y_d);
  });

  cache->restr_f->MultTranspose(rhs_faces_x, faces_dudx);
  cache->restr_f->MultTranspose(rhs_faces_y, faces_dudy);

  // on-device?
  pdudx += faces_dudx;
  pdudy += faces_dudy;

}

void DGSEMNonlinearForm::GradOperator(const mfem::Vector &u,
                                      std::vector<mfem::Vector *> &grad_u) const
{
  const int dim = cache->dim;
  const Vector &pu = Prolongate(u);

  std::vector<mfem::Vector *> p_grad_(dim);
  if (P)
    {
      const int psize = P->Height();
      if(this->grad_aux_.size() != dim){
        grad_aux_.resize(dim);
      }
      for(int idim = 0;idim < dim;idim++){
        grad_aux_[idim].SetSize(psize);
        p_grad_[idim] = &grad_aux_[idim];
      }
    }
  std::vector<mfem::Vector *> &p_grad_u = P ? p_grad_ : grad_u;

  MFEM_ASSERT(p_grad_u.size() == dim, "Size mismatch for gradient storage");
  MFEM_ASSERT(grad_u.size() == dim, "Size mismatch for gradient storage");

  GradOperatorVolume(pu, p_grad_u);
 
  GradOperatorInteriorFaces(pu, p_grad_u);

  GradOperatorBoundaryFaces(pu, p_grad_u);

  if (Serial())
    {
      if (cP)
        {
          for(int idim = 0;idim < dim;idim++){
            cP->MultTranspose(*p_grad_u[idim], *grad_u[idim]);
          }
        }
    }
  else
    {
      if(P){
        for(int idim = 0;idim < dim;idim++){
          P->MultTranspose(*p_grad_u[idim], *grad_u[idim]);
        }
      }
    }

  const int N = ess_tdof_list.Size();
  const auto idx = ess_tdof_list.Read();
  for(int idim = 0;idim < dim;idim++){
    auto gradu_dim_d = grad_u[idim]->ReadWrite();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { gradu_dim_d[idx[i]] = 0.0; });
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

// Then the device port/infrastructure for that top level inviscid boundary faces is:
real_t DGSEMNonlinearForm::MultCNS_BoundaryFaces(const mfem::Vector &pu,
                                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                                 mfem::Vector &pdudt) const
{
  auto dc = device_cache;
  const int dim = dc.dim;
  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int face_size = nfp * neq;
  const int restr_size = cache->restr_b->Height();
  const int nfaces_restr = restr_size / face_size;
  const int norm_size = nfp * dc.dim;
  const int npoints_bnd = nfaces_restr * nfp;

  mfem::Vector rhs_faces(restr_size);
  mfem::Vector faces_dudt(pdudt.Size());
  // mfem::Vector u_faces(restr_size);
  bnd_u.SetSize(restr_size);
  const real_t *grad_prim_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  // const real_t *face_grad_prim[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  bnd_grad_prim.resize(dim);
  for(int idim = 0;idim < dim;idim++){
    bnd_grad_prim[idim].SetSize(restr_size);
    bnd_grad_prim[idim].UseDevice();
    cache->restr_b->Mult(*p_grad_prim[idim], bnd_grad_prim[idim]);
    grad_prim_d[idim] = bnd_grad_prim[idim].Read();
  }

  faces_dudt.UseDevice();
  rhs_faces.UseDevice();
  bnd_u.UseDevice();

  // If zeroed before accumulation, do it explicitly on device:
  // Potentially, this is not needed at all since I think we overwrite everything
  {
    real_t *rd = rhs_faces.Write();
    mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i)
    { rd[i] = real_t(0);});
    real_t *fd = faces_dudt.Write();
    mfem::forall(faces_dudt.Size(), [=] MFEM_HOST_DEVICE (int i)
    { fd[i] = real_t(0);});
  }

  cache->restr_b->Mult(pu, bnd_u);

  const real_t *u_d = bnd_u.Read();
  real_t *rhs_d = rhs_faces.Write();

  const real_t *nor_d   = dc.bnd_nor_d;      // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.bnd_wt_d; // size nfaces*nfp
  const int *bnd_marker_index_d = dc.bnd_marker_index_d;
  // const int *bnd_attr = dc.bnd_attr_d;
  // const int *bnd_marker_to_bc_descr_d = dc.bnd_marker_to_bc_descr_d;
  real_t *ws_d = dc.bndWaveSpeed_d;

  mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
  {
    const int f = p / nfp;
    const int fp = p % nfp;

    int bnd_face_marker_index = bnd_marker_index_d[f];
    if(bnd_face_marker_index < 0){
      ws_d[p] = 0.0;
      return;
    }
    //    int bc_index = bnd_marker_to_bc_descr_d[bnd_face_marker_index];
    int bc_index = bnd_face_marker_index; // no mapping atm
    if(bc_index < 0){
      ws_d[p] = 0.0;
      return;
    }
    const Prandtl::BCDescriptor &bc = dc.bc_descr_d[bc_index];
    if (bc.type == int(Prandtl::BCType::Invalid))
      {
        ws_d[p] = 0.0;
        return;
      }

    const int face_offset = f * face_size;
    const int n_offset = f * norm_size;
    const int w_offset = f * nfp;

    const real_t *u_face_d = u_d + face_offset;
    real_t *rhs_face_d = rhs_d + face_offset;
    const real_t *nor_face_d = nor_d + n_offset;
    const real_t *w_minus_d = inv1_d + w_offset;
    const real_t *nor_point = nor_face_d + fp*dim;
    real_t scale = -w_minus_d[fp];
    // #ifdef AXISYMMETRIC
    // NOTE: axisymmetric not ready for device yet
    // scale *= rad_face[fp];
    // #else
    // #error "Axisymmetric boundary device path not implemented yet."
    // #endif
    real_t state1[Prandtl::MAXEQ];
    real_t fluxN[Prandtl::MAXEQ];
    real_t gradPrim_x[Prandtl::MAXEQ];
    real_t gradPrim_y[Prandtl::MAXEQ];
    real_t gradPrim_z[Prandtl::MAXEQ];
    const real_t *dprim_face_x = (dim > 0) ? grad_prim_d[0] + face_offset : nullptr;
    const real_t *dprim_face_y = (dim > 1) ? grad_prim_d[1] + face_offset : nullptr;
    const real_t *dprim_face_z = (dim > 2) ? grad_prim_d[2] + face_offset : nullptr;
    Prandtl::Kernels::el_gather_grad_state(dprim_face_x, dprim_face_y, dprim_face_z,
                                           dim, nfp, neq, fp, gradPrim_x, gradPrim_y,
                                           gradPrim_z);
    Prandtl::Kernels::el_gather_state(u_face_d, nfp, neq, fp, state1);
    
    const real_t ws = \
      Prandtl::BC::ApplyViscousBoundaryCondition(dc, bc, state1, gradPrim_x, gradPrim_y,
                                                 gradPrim_z, nor_point, fluxN);
    Prandtl::Kernels::el_scatter_add(fluxN, nfp, neq, fp, scale, rhs_face_d);
    ws_d[p] = ws;
  });

  cache->restr_b->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt; // on device? (likely yes) 

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache->bndWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int p = 0;p < npoints_bnd;p++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[p]);
    }

  return max_char_speed_facial;
}

real_t DGSEMNonlinearForm::MultCNS_Volume(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                                          mfem::Vector &pdudt) const
{
  // ScopedTimer timer("AssembleCNSRHS_Volume");
  // Copy the device cache so that it is not member data
  auto dc = device_cache;
  const int dim = dc.dim;

  const int restr_size = cache->restr_v->Height();
  // This block is executed by the host
  // mfem::Vector Ue(cache->restr_v->Height());
  mfem::Vector dUe(restr_size);
  if(vol_u.Size() != restr_size){
    vol_u.SetSize(restr_size);
    vol_u.UseDevice();
    vol_grad_prim.resize(dim);
    for(int idim = 0;idim < dim;idim++){
      vol_grad_prim[idim].SetSize(restr_size);
      vol_grad_prim[idim].UseDevice();
    }
  }

  cache->restr_v->Mult(pu, vol_u);
  for(int idim = 0;idim < dim;idim++){
    cache->restr_v->Mult(*p_grad_prim[idim], vol_grad_prim[idim]);
  }

  // Zero the RHS array on-device
  {
    real_t *d = dUe.Write();
    mfem::forall(dUe.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  // Set up the read-only pointers for restr inputs
  const real_t *Ue_d = vol_u.Read();
  const real_t *gradPrim_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  for(int idim = 0;idim < dim;idim++){
    gradPrim_d[idim] = vol_grad_prim[idim].Read();
  }
  // Write-only for RHS
  real_t *dUe_d = dUe.Write();

  // Device cache parameters
  const int ne = dc.num_elements;
  const int ndof = dc.ndof_scalar_el;
  const int neq = dc.num_equations;

#ifdef SUBCELL_FV_BLENDING
  const int Np_x = dc.Np_x;
  const int Np_y = dc.Np_y;
  const int Np_z = dc.Np_z;
  const int npe = Np_x * Np_y * Np_z;
  const int ndofe = npe * neq;
  const int npe_metric_xi = (Np_x + 1)*Np_y*Np_z;
  const int npe_metric_eta = Np_x*(Np_y + 1)*Np_z;
  const int npe_metric_zeta = Np_x * Np_y * (Np_z + 1);
  const real_t *metric_xi_d = dc.subcell_metric_xi_d;
  const real_t *metric_eta_d = (dim > 1 ? dc.subcell_metric_eta_d : nullptr);
  const real_t *metric_zeta_d = (dim > 2 ? dc.subcell_metric_zeta_d : nullptr);

  mfem::Vector dUfv(cache->restr_v->Height());
  dUfv.UseDevice();
  real_t *dUfv_d = dUfv.Write();
  // zero the array on-device
  {
    real_t *d = dUfv_d;
    mfem::forall(dUfv.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  const real_t *alpha_d = cache->alpha.Read();
#endif

  // Derived parameters
  const int metric_stride = ndof * dim * dim;
  const int jac_stride    = ndof;
  const int estride = ndof*neq;
  
  // Device cache data/arrays
  const int *elem_attr_d = dc.elem_attr_d;
  const int *attr_marker_d = dc.attr_marker_d;
  const real_t *elJac_d = dc.elJac_d;
  const real_t *elMetric_d = dc.elMetric_d;

  real_t *ws_d = dc.elWaveSpeed_d;

  // Inside the FORALL below, executed on device
  mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
  {
    
    const real_t *jac_el    = elJac_d    + e * jac_stride;
    const real_t *metric_el = elMetric_d + e * metric_stride;

    const int attr = elem_attr_d[e];
    if (attr_marker_d[attr-1] == 0) {
      ws_d[e] = 0.0;
      return;
    }

    // Element-specific inputs and outputs
    const int eoff = e * estride;
    const real_t *u_el = Ue_d + eoff;
    real_t *du_el = dUe_d + eoff;

    real_t cs_el = \
      DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                   jac_el, metric_el, du_el);
#ifdef SUBCELL_FV_BLENDING
    real_t alpha_fv = alpha_d[e];
    if(alpha_fv > 1e-16){
      real_t alpha_dg = (1.0 - alpha_fv);
      real_t *du_fv = dUfv_d + eoff;
      const real_t *el_metric_xi = metric_xi_d + e * npe_metric_xi * dim;
      const real_t *el_metric_eta = (dim > 1 ? metric_eta_d + e * npe_metric_eta * dim :
                                     nullptr);
      const real_t *el_metric_zeta = (dim > 2 ? metric_zeta_d + e * npe_metric_zeta * dim :
                                      nullptr);
      const real_t cs_fv =                                              \
        DGSEMIntegrator::ComputeFVFluxesKernel(dc, u_el, jac_el, el_metric_xi, el_metric_eta, el_metric_zeta, du_fv);
      
      for(int ipt = 0;ipt < estride;ipt++){
        du_el[ipt] = alpha_dg * du_el[ipt] + alpha_fv * du_fv[ipt];
      }

      cs_el = Kernels::rmax(cs_el, cs_fv);
    }
#endif
    ws_d[e] = cs_el;

    // Inviscid part is done: dUe currrently holds the inviscid RHS
    // Host code mixes inviscid and viscous assembly, we need separate.
    // Call the Viscous Assembly routine
    const real_t *grad_prim_el[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      grad_prim_el[idim] = gradPrim_d[idim] + eoff;
    }

    DGSEMIntegrator::AssembleViscousElementVolumeKernel(dc, u_el, jac_el, metric_el,
                                                        grad_prim_el[0], grad_prim_el[1],
                                                        grad_prim_el[2], du_el);

  });

  // The rest is identical to Euler operator
  // Scatter RHS back to storage
  cache->restr_v->AddMultTranspose(dUe, pdudt);

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache->elWaveSpeed.HostRead();
  real_t max_char_speed = 0.0;
  for(int e = 0;e < cache->num_elements;e++)
    {
      max_char_speed = std::max(max_char_speed, ws[e]);
    }

  return max_char_speed;
}

// Assemble volume part of RHS for all elements
// This routine will eventually just replace the original code
// once the disabled/broken features can be reimplemented
// This is the version that is device-ready and currently used by Prandtl.
// NOTE:
//  - No axisymmetry (broken in device version of MULT)
//  - No subcell blending (broken in device version of MULT)
real_t DGSEMNonlinearForm::MultVolumeInviscid(const Vector &pu, Vector &pdudt) const
{
  // ScopedTimer timer("MultVolumeInviscidDevice");

  // This block is executed by the host
  mfem::Vector Ue(cache->restr_v->Height());
  mfem::Vector dUe(cache->restr_v->Height());

  Ue.UseDevice();
  dUe.UseDevice();

  cache->restr_v->Mult(pu, Ue);
  
  // Zero the array on-device
  {
    real_t *d = dUe.Write();
    mfem::forall(dUe.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  const real_t *Ue_d = Ue.Read();
  real_t *dUe_d = dUe.Write();

  // Copy the device cache so that it is not member data
  auto dc = device_cache;

  // Device cache parameters
  const int dim = dc.dim;
  const int ne = dc.num_elements;
  const int ndof = dc.ndof_scalar_el;
  const int neq = dc.num_equations;

#ifdef SUBCELL_FV_BLENDING
  const int Np_x = dc.Np_x;
  const int Np_y = dc.Np_y;
  const int Np_z = dc.Np_z;
  const int npe = Np_x * Np_y * Np_z;
  const int ndofe = npe * neq;
  const int npe_metric_xi = (Np_x + 1)*Np_y*Np_z;
  const int npe_metric_eta = Np_x*(Np_y + 1)*Np_z;
  const int npe_metric_zeta = Np_x * Np_y * (Np_z + 1);
  const real_t *metric_xi_d = dc.subcell_metric_xi_d;
  const real_t *metric_eta_d = (dim > 1 ? dc.subcell_metric_eta_d : nullptr);
  const real_t *metric_zeta_d = (dim > 2 ? dc.subcell_metric_zeta_d : nullptr);

  mfem::Vector dUfv(cache->restr_v->Height());
  dUfv.UseDevice();
  real_t *dUfv_d = dUfv.Write();
  // zero the array on-device
  {
    real_t *d = dUfv_d;
    mfem::forall(dUfv.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  const real_t *alpha_d = cache->alpha.Read();
#endif

  // Derived parameters
  const int metric_stride = ndof * dim * dim;
  const int jac_stride    = ndof;
  const int estride = ndof*neq;
  
  // Device cache data/arrays
  const int *elem_attr_d = dc.elem_attr_d;
  const int *attr_marker_d = dc.attr_marker_d;
  const real_t *elJac_d = dc.elJac_d;
  const real_t *elMetric_d = dc.elMetric_d;

  real_t *ws_d = dc.elWaveSpeed_d;

  // Inside the FORALL below, executed on device
  mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
  {
    
    const real_t *jac_el    = elJac_d    + e * jac_stride;
    const real_t *metric_el = elMetric_d + e * metric_stride;

    const int attr = elem_attr_d[e];
    if (attr_marker_d[attr-1] == 0) {
      ws_d[e] = 0.0;
      return;
    }

    const int eoff = e * estride;
    const real_t *u_el = Ue_d + eoff;
    real_t *du_el = dUe_d + eoff;

    real_t cs_el = \
      DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                   jac_el, metric_el, du_el);
#ifdef SUBCELL_FV_BLENDING
    real_t alpha_fv = alpha_d[e];
    if(alpha_fv > 1e-16){
      real_t alpha_inv = (1.0 - alpha_fv);
      real_t *du_fv = dUfv_d + eoff;
      const real_t *el_metric_xi = metric_xi_d + e * npe_metric_xi * dim;
      const real_t *el_metric_eta = (dim > 1 ? metric_eta_d + e * npe_metric_eta * dim :
                                     nullptr);
      const real_t *el_metric_zeta = (dim > 2 ? metric_zeta_d + e * npe_metric_zeta * dim :
                                      nullptr);
      const real_t cs_fv =                                              \
        DGSEMIntegrator::ComputeFVFluxesKernel(dc, u_el, jac_el, el_metric_xi, el_metric_eta, el_metric_zeta, du_fv);
      
      for(int ipt = 0;ipt < estride;ipt++){
        du_el[ipt] = alpha_inv * du_el[ipt] + alpha_fv * du_fv[ipt];
      }

      cs_el = Kernels::rmax(cs_el, cs_fv);
    }
#endif

    ws_d[e] = cs_el;

  });

  // Scatter RHS back to storage
  cache->restr_v->AddMultTranspose(dUe, pdudt);

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache->elWaveSpeed.HostRead();
  real_t max_char_speed = 0.0;
  for(int e = 0;e < cache->num_elements;e++)
    {
      max_char_speed = std::max(max_char_speed, ws[e]);
    }

  return max_char_speed;
}

real_t DGSEMNonlinearForm::MultInteriorFacesInviscid(const Vector &pu, Vector &pdudt) const
{
  //  ScopedTimer timer("MultInteriorFacesInviscidDevice");
  auto dc = device_cache;
  const int dim = dc.dim;
  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int nfaces = cache->restr_f->Height() / (nfp * neq * 2); // (+/-)
  const int face_stride = 2 * nfp * neq;
  const int side_stride = nfp * neq;
  const int face_size = 2*nfp*neq;
  const int norm_size = nfp*dim;
  
  mfem::Vector u_faces(cache->restr_f->Height());
  mfem::Vector rhs_faces(cache->restr_f->Height());
  mfem::Vector faces_dudt(pdudt);
  faces_dudt.UseDevice();
  rhs_faces.UseDevice();
  u_faces.UseDevice();

  // If zeroed before accumulation, do it explicitly on device:
  // Potentially, this is not needed at all since I think we overwrite everything
  {
    real_t *d = rhs_faces.Write();
    mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  cache->restr_f->Mult(pu, u_faces);

  const real_t *u_d = u_faces.Read();
  real_t *rhs_d = rhs_faces.Write();

  const real_t *nor_d   = dc.nor_d;      // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.fw_minus_d; // size nfaces*nfp
  const real_t *inv2_d  = dc.fw_plus_d;  // size nfaces*nfp

  real_t *ws_d = dc.ifWaveSpeed_d;

  mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int i)
  {
    const int face_offset = i*face_size;
    const int n_offset = i*norm_size;
    const int w_offset = i*nfp;

    const real_t *u_face_d = u_d + face_offset;
    real_t *rhs_face_d = rhs_d + face_offset;
    const real_t *nor_face_d = nor_d + n_offset;
    const real_t *w_minus_d = inv1_d + w_offset;
    const real_t *w_plus_d = inv2_d + w_offset;
    
    real_t ws = DGSEMIntegrator::AssembleElementFaceKernel(dc, u_face_d, nor_face_d,
                                                           w_minus_d, w_plus_d, rhs_face_d);
    ws_d[i] = ws;
    
  });

  cache->restr_f->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt; // on device? 

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache->ifWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int f = 0;f < cache->num_interior_faces;f++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[f]);
    }

  return max_char_speed_facial;
}


real_t DGSEMNonlinearForm::MultBndFacesInviscid(const Vector &pu, Vector &pdudt) const
{
  //  ScopedTimer timer("MultBndFacesInviscidDevice");
  auto dc = device_cache;
  const int dim = dc.dim;
  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int face_size = nfp * neq;
  const int nfaces_restr = cache->restr_b->Height() / face_size;
  const int norm_size = nfp * dc.dim;
  const int npoints_bnd = nfaces_restr * nfp;

  mfem::Vector u_faces(cache->restr_b->Height());
  mfem::Vector rhs_faces(cache->restr_b->Height());
  mfem::Vector faces_dudt(pdudt.Size());

  faces_dudt.UseDevice();
  rhs_faces.UseDevice();
  u_faces.UseDevice();

  // If zeroed before accumulation, do it explicitly on device:
  // Potentially, this is not needed at all since I think we overwrite everything
  {
    real_t *rd = rhs_faces.Write();
    mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i)
    { rd[i] = real_t(0);});
    real_t *fd = faces_dudt.Write();
    mfem::forall(faces_dudt.Size(), [=] MFEM_HOST_DEVICE (int i)
    { fd[i] = real_t(0);});
  }

  cache->restr_b->Mult(pu, u_faces);

  const real_t *u_d = u_faces.Read();
  real_t *rhs_d = rhs_faces.Write();

  const real_t *nor_d   = dc.bnd_nor_d;      // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.bnd_wt_d; // size nfaces*nfp
  const int *bnd_marker_index_d = dc.bnd_marker_index_d;
  // const int *bnd_attr = dc.bnd_attr_d;
  // const int *bnd_marker_to_bc_descr_d = dc.bnd_marker_to_bc_descr_d;
  real_t *ws_d = dc.bndWaveSpeed_d;

  mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
  {
    const int f = p / nfp;
    const int fp = p % nfp;

    int bnd_face_marker_index = bnd_marker_index_d[f];
    if(bnd_face_marker_index < 0){
      ws_d[p] = 0.0;
      return;
    }
    //    int bc_index = bnd_marker_to_bc_descr_d[bnd_face_marker_index];
    int bc_index = bnd_face_marker_index; // no mapping atm
    if(bc_index < 0){
      ws_d[p] = 0.0;
      return;
    }
    const Prandtl::BCDescriptor &bc = dc.bc_descr_d[bc_index];
    if (bc.type == int(Prandtl::BCType::Invalid))
      {
        ws_d[p] = 0.0;
        return;
      }

    const int face_offset = f * face_size;
    const int n_offset = f * norm_size;
    const int w_offset = f * nfp;

    const real_t *u_face_d = u_d + face_offset;
    real_t *rhs_face_d = rhs_d + face_offset;
    const real_t *nor_face_d = nor_d + n_offset;
    const real_t *w_minus_d = inv1_d + w_offset;
    const real_t *nor_point = nor_face_d + fp*dim;
    real_t scale = -w_minus_d[fp];
    // #ifdef AXISYMMETRIC
    // NOTE: axisymmetric not ready for device yet
    // scale *= rad_face[fp];
    // #else
    // #error "Axisymmetric boundary device path not implemented yet."
    // #endif
    real_t state1[Prandtl::MAXEQ];
    real_t fluxN[Prandtl::MAXEQ];

    Prandtl::Kernels::el_gather_state(u_face_d, nfp, neq, fp, state1);
    const real_t ws = \
      Prandtl::BC::ApplyBoundaryConditionInviscid(dc, bc, state1,
                                                  nor_point, fluxN);
    Prandtl::Kernels::el_scatter_add(fluxN, nfp, neq, fp, scale, rhs_face_d);
    ws_d[p] = ws;

  });

  cache->restr_b->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt; // on device? (likely yes) 

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache->bndWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int p = 0;p < npoints_bnd;p++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[p]);
    }

  return max_char_speed_facial;
}


// Top level MULT for inviscid cases, called from DGSEMOperator
real_t DGSEMNonlinearForm::MultEuler(const Vector &u, Vector &dudt) const
{
  // ScopedTimer timer("MultInviscid");
  const Vector &pu = Prolongate(u);
  if (P)
    {
      aux2.SetSize(P->Height());
    }

  Vector &pdudt = P ? aux2 : dudt;
  pdudt = 0.0;

  real_t max_char_speed = 0.0;
  // This step overwrites contents of pdudt
  max_char_speed = MultVolumeInviscid(pu, pdudt);
  // std::cout << "Volume wavespeed: " << max_char_speed << std::endl;

  real_t max_char_speed_facial = 0.0;
  max_char_speed_facial = MultInteriorFacesInviscid(pu, pdudt);
  // std::cout << "Facial wavespeed: " << max_char_speed_facial << std::endl;
  max_char_speed = std::max(max_char_speed, max_char_speed_facial);

  real_t max_char_speed_bnd = 0.0;
  max_char_speed_bnd = MultBndFacesInviscid(pu, pdudt);
  // max_char_speed_bnd = MultBndFacesInviscidHost(pu, pdudt);
  // std::cout << "Boundary wavespeed: " << max_char_speed_bnd << std::endl;
  max_char_speed = std::max(max_char_speed, max_char_speed_bnd);

  if (Serial())
    {
      if(cP) cP->MultTranspose(pdudt, dudt);
    }
  else
    {
      if(P) P->MultTranspose(aux2, dudt);
    }

  const int N = ess_tdof_list.Size();
  const auto idx = ess_tdof_list.Read();
  auto DU_RW = dudt.ReadWrite();
  mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

  return max_char_speed;
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

real_t DGSEMNonlinearForm::MultCNS(const mfem::Vector &u, const std::vector<mfem::Vector *> &grad_prim,
                                   mfem::Vector &dudt) const
{
  const int dim = cache->dim;
  const Vector &pu = Prolongate(u);

  if (P)
    {
      const int psize = P->Height();
      rhs_aux_.SetSize(psize);
      if(grad_aux_.size() != dim){
        grad_aux_.resize(dim);
      }
      for(int idim = 0;idim < dim;idim++){
        grad_aux_[idim].SetSize(psize);
        P->Mult(*grad_prim[idim], grad_aux_[idim]);
      }
      // aux2.SetSize(P->Height());
      // aux2_x.SetSize(P->Height());
      // aux2_y.SetSize(P->Height());

      // P->Mult(*grad_prim[0], aux2_x);
      // P->Mult(*grad_prim[1], aux2_y);
    }

  Vector &pdudt = P ? rhs_aux_ : dudt;
  // const Vector &pdudx = P ? aux2_x : dudx;
  // const Vector &pdudy = P ? aux2_y : dudy;
  std::vector<mfem::Vector *> pGradPrim(dim);
  for(int idim = 0;idim < dim;idim++){
    pGradPrim[idim] = &grad_aux_[idim];
  }
  //  pGradPrim[1] = &aux2_y;

  pdudt = 0.0;
  real_t max_char_speed = MultCNS_Volume(pu, pGradPrim, pdudt);
  // MultCNS_Volume_Host(pu, pGradPrim, pdudt);
  // MultCNS_InteriorFaces_Host(pu, pGradPrim, pdudt);
  real_t max_char_speed_faces = MultCNS_InteriorFaces(pu, pGradPrim, pdudt);
  max_char_speed = std::max(max_char_speed, max_char_speed_faces);

  real_t max_char_speed_bnd = MultCNS_BoundaryFaces(pu, pGradPrim, pdudt);
  // MultCNS_BoundaryFaces_Host(pu, pGradPrim, pdudt);
  max_char_speed = std::max(max_char_speed, max_char_speed_bnd);
  
  if (Serial())
    {
      if (cP)
        {
          cP->MultTranspose(pdudt, dudt);
        }

    }
  else
    {
      P->MultTranspose(pdudt, dudt);
    }

  const int N = ess_tdof_list.Size();
  const auto idx = ess_tdof_list.Read();
  auto DU_RW = dudt.ReadWrite();
  mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

  return max_char_speed;
}

real_t DGSEMNonlinearForm::MultCNS_InteriorFaces(const mfem::Vector &pu,
                                                 const std::vector<mfem::Vector *> &p_grad_prim,
                                                 mfem::Vector &pdudt) const
{
  //  ScopedTimer timer("MultInteriorFacesInviscidDevice");
  auto dc = device_cache;
  const int dim = dc.dim;
  const int neq = dc.num_equations;
  const int nfp = dc.num_face_points;
  const int nfaces = cache->restr_f->Height() / (nfp * neq * 2); // (+/-)
  const int face_stride = 2 * nfp * neq;
  const int side_stride = nfp * neq;
  const int face_size = 2*nfp*neq;
  const int norm_size = nfp*dim;

  const int restr_size = cache->restr_f->Height();
  int_u.SetSize(restr_size);
  int_u.UseDevice();
  
  // mfem::Vector u_faces(restr_size);
  mfem::Vector rhs_faces(restr_size);
  mfem::Vector faces_dudt(pdudt);
  const real_t *grad_prim_d[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  // const real_t *face_grad_prim[Prandtl::MAXDIM] = {nullptr, nullptr, nullptr};
  if(int_grad_prim.size() != dim){
    int_grad_prim.resize(dim);
  }
  for(int idim = 0;idim < dim;idim++){
    int_grad_prim[idim].SetSize(restr_size);
    int_grad_prim[idim].UseDevice();
    cache->restr_f->Mult(*p_grad_prim[idim], int_grad_prim[idim]);
    grad_prim_d[idim] = int_grad_prim[idim].Read();
  }
  faces_dudt.UseDevice();
  rhs_faces.UseDevice();
  // u_faces.UseDevice();
  
  // If zeroed before accumulation, do it explicitly on device:
  // Potentially, this is not needed at all since I think we overwrite everything
  {
    real_t *d = rhs_faces.Write();
    mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = real_t(0); });
  }

  cache->restr_f->Mult(pu, int_u);

  const real_t *u_d = int_u.Read();
  real_t *rhs_d = rhs_faces.Write();
  const real_t *nor_d   = dc.nor_d;      // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.fw_minus_d; // size nfaces*nfp
  const real_t *inv2_d  = dc.fw_plus_d;  // size nfaces*nfp

  real_t *ws_d = dc.ifWaveSpeed_d;

  mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int i)
  {
    const int face_offset = i*face_size;
    const int n_offset = i*norm_size;
    const int w_offset = i*nfp;

    const real_t *u_face_d = u_d + face_offset;
    real_t *rhs_face_d = rhs_d + face_offset;
    const real_t *nor_face_d = nor_d + n_offset;
    const real_t *w_minus_d = inv1_d + w_offset;
    const real_t *w_plus_d = inv2_d + w_offset;
    const real_t *dprim_face_x = (dim > 0) ? grad_prim_d[0] + face_offset : nullptr;
    const real_t *dprim_face_y = (dim > 1) ? grad_prim_d[1] + face_offset : nullptr;
    const real_t *dprim_face_z = (dim > 2) ? grad_prim_d[2] + face_offset : nullptr;
    // for(int idim = 0;idim < dim;idim++){
    //   face_grad_prim[idim] = grad_prim_d[idim] + face_offset;
    // }

    // Call one fused kernel for inviscid and viscous facial terms
    real_t ws = DGSEMIntegrator::AssembleViscousElementFaceKernel(dc, u_face_d, nor_face_d,
                                                                  w_minus_d, w_plus_d,
                                                                  dprim_face_x,
                                                                  dprim_face_y,
                                                                  dprim_face_z,
                                                                  rhs_face_d);
    ws_d[i] = ws;
  });

  cache->restr_f->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt; // on device? 

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache->ifWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int f = 0;f < cache->num_interior_faces;f++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[f]);
    }

  return max_char_speed_facial;
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
