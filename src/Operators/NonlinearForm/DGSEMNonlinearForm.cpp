#include "DGSEMNonlinearForm.hpp"
#include "timer.hpp"

namespace Prandtl
{

void OutputOperatorCache(const DGSEMNonlinearForm::OperatorCache &cache)
{
  std::cout << "Operator Cache:" << std::endl
            << "p = " << cache.p << std::endl
            << "dim = " << cache.dim << std::endl
            << "num_elements = " << cache.num_elements << std::endl
            << "Np,Np_x,Np_y,Np_z = " << cache.Np << "," << cache.Np_x
            << "," << cache.Np_y << "," << cache.Np_z << std::endl
            << "num_face_points = " << cache.num_face_points << std::endl
            << "num_attr = " << cache.num_attr << std::endl
            << "ndof_scalar_el = " << cache.ndof_scalar_el << std::endl
            << "num_interior_faces = " << cache.num_interior_faces << std::endl;
  MFEM_VERIFY(cache.ir, "IR is not set");
  MFEM_VERIFY(cache.ir_face, "Face IR not set");
  MFEM_VERIFY(cache.ir_vol, "Volume IR not set");
  MFEM_VERIFY(cache.restr_v, "Volume Restriction not set");
  MFEM_VERIFY(cache.restr_f, "Facial Restriction not set");
  MFEM_VERIFY(cache.ndof_scalar_el == cache.Np_x*cache.Np_y*cache.Np_z,
              "Element dof count not equal to num quadrature points.");
  int ds_size = cache.elem_attr.Size();
  MFEM_VERIFY(ds_size > 0, "Elem attr not set");
  ds_size = cache.attr_marker.Size();
  MFEM_VERIFY(ds_size > 0, "Attr markers not set");
  ds_size = cache.dnfi_marker.Size();
  MFEM_VERIFY(ds_size > 0, "dnfi markers not set");
  ds_size = cache.elWaveSpeed.Size();
  MFEM_VERIFY(ds_size == cache.num_elements, "Element wavespeeds missized.");
  ds_size = cache.bndWaveSpeed.Size();
  ds_size = cache.elJac.Size();
  MFEM_VERIFY(ds_size > 0, "Element Jacobians not set");
  ds_size = cache.elMetric.Size();
  MFEM_VERIFY(ds_size > 0, "Element Metrics not set");
  ds_size = cache.D.Size();
  MFEM_VERIFY(ds_size > 0, "Deriv operator not set");
  ds_size = cache.Dhat2.Size();
  MFEM_VERIFY(ds_size > 0, "Dhat2 operator not set");
  ds_size = cache.face_normals.Size();
  MFEM_VERIFY(ds_size == cache.num_face_points*cache.num_interior_faces*cache.dim,
              "Inapropriately sized face normals");
  ds_size = cache.face_wt_minus.Size();
  ds_size = cache.face_wt_plus.Size();
  MFEM_VERIFY(ds_size > 0, "Face weights not set.");
}

  DGSEMNonlinearForm::DGSEMNonlinearForm(ParFiniteElementSpace *pf)
    : ParNonlinearForm(pf)
  {
    GRAD_X.MakeRef(pf, NULL);
    GRAD_Y.MakeRef(pf, NULL);
    GRAD_Z.MakeRef(pf, NULL);
  }

  void DGSEMNonlinearForm::AssembleFaceGeomCacheInterior()
  {
    auto *mesh = fes->GetMesh();
    auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    cache.fqs_int = new mfem::FaceQuadratureSpace(*mesh, *cache.ir_face, mfem::FaceType::Interior);
    MFEM_VERIFY(pfes, "need ParFiniteElementSpace");
    
    const int dim = mesh->Dimension();
    const int neq = pfes->GetVDim();
    const int nfp = cache.ir_face->GetNPoints();

    auto &int_faces = mesh->GetFaceIndices(mfem::FaceType::Interior);
    BuildFaceLists();
    const int nfaces = int_faces.Size();

    cache.inv_fp_map.SetSize(nfaces * nfp);    
    for (int face_slot = 0; face_slot < nfaces; ++face_slot)
      {
        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            int fp_perm = cache.fqs_int->GetPermutedIndex(face_slot, fp_restr);
            cache.inv_fp_map[face_slot*nfp + fp_perm] = fp_restr;
          }
      }

    double *nor_d  = cache.face_normals.HostWrite();
    double *inv1_d = cache.face_wt_minus.HostWrite();
    double *inv2_d = cache.face_wt_plus.HostWrite();
    const real_t w0 = cache.ir->IntPoint(0).weight;
    
    auto store = [&](int fslot, int fp, const mfem::Vector &nor,
                     double inv_wJ1, double inv_wJ2)
    {
      const int nbase = (fslot * nfp + fp) * dim;
      for (int d = 0; d < dim; ++d) { nor_d[nbase + d] = nor(d); }
      inv1_d[fslot * nfp + fp] = inv_wJ1;
      inv2_d[fslot * nfp + fp] = inv_wJ2;
    };
    
    mfem::Vector nor(dim);
    const int num_elements_pmesh = pmesh->GetNE();
    for (int fslot = 0; fslot < nfaces; ++fslot)
      {
        const int face_id = int_faces[fslot];
        if(cache.mesh_face_is_shared[face_id]){ // Do shared face caching
          auto *tr = pmesh->GetSharedFaceTransformationsByLocalIndex(face_id, true);
          MFEM_VERIFY(tr, "expected shared face");
          bool face_is_flipped = false;
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = MapFp(fslot, fp_restr);// <-- critical
              if (fp_geom != fp_restr){
                face_is_flipped = true;
              }
            }
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = MapFp(fslot, fp_restr);// <-- critical
              const mfem::IntegrationPoint &ip = cache.ir_face->IntPoint(fp_geom);
              tr->SetAllIntPoints(&ip);
              
              const double J1 = tr->GetElement1Transformation().Weight();
              const double J2 = tr->GetElement2Transformation().Weight();
              
              if (dim == 1) { nor(0) = (tr->GetElement1IntPoint().x - 0.5)*2.0; }
              else          { mfem::CalcOrtho(tr->Jacobian(), nor); }
              
              //const real_t fac = face_is_flipped ? -1.0 : 1.0;
              const real_t fac1 = 1.0;
              const real_t fac2 = 0.0;
              store(fslot, fp_restr, nor, fac1/(w0*J1), fac2/(w0*J2));
            }
        } else { // local internal face
          auto *tr = mesh->GetInteriorFaceTransformations(face_id);
          MFEM_VERIFY(tr, "expected interior face");
          bool face_is_flipped = false;
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = MapFp(fslot, fp_restr);// <-- critical
              if (fp_geom != fp_restr){
                face_is_flipped = true;
              }
            }
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = MapFp(fslot, fp_restr);// <-- critical
              const mfem::IntegrationPoint &ip = cache.ir_face->IntPoint(fp_geom);
              tr->SetAllIntPoints(&ip);
              
              const double J1 = tr->GetElement1Transformation().Weight();
              const double J2 = tr->GetElement2Transformation().Weight();
              
              if (dim == 1) { nor(0) = (tr->GetElement1IntPoint().x - 0.5)*2.0; }
              else          { mfem::CalcOrtho(tr->Jacobian(), nor); }
              
              //const real_t fac = face_is_flipped ? -1.0 : 1.0;
              const real_t fac = 1.0;
              store(fslot, fp_restr, nor, fac/(w0*J1), fac/(w0*J2));
            }
        } // Internal face processing
      }
  }

  void DGSEMNonlinearForm::GetDeviceCache(Prandtl::DGSEMDeviceCache &dgsem_device_cache)
  {
    dgsem_device_cache.ndof_scalar_el = cache.ndof_scalar_el;
    dgsem_device_cache.num_attr = cache.num_attr;
    dgsem_device_cache.attr_marker_d = cache.attr_marker.Read();
    dgsem_device_cache.elem_attr_d = cache.elem_attr.Read();
    dgsem_device_cache.num_face_points = cache.num_face_points;
    dgsem_device_cache.p = cache.p;
    dgsem_device_cache.dim = cache.dim;
    dgsem_device_cache.Np = cache.Np;
    dgsem_device_cache.Np_x = cache.Np_x;
    dgsem_device_cache.Np_y = cache.Np_y;
    dgsem_device_cache.Np_z = cache.Np_z;
    dgsem_device_cache.num_elements = cache.num_elements;
    dgsem_device_cache.num_equations = cache.num_equations;
    dgsem_device_cache.elJac_d = cache.elJac.Read();
    dgsem_device_cache.elMetric_d = cache.elMetric.Read();
    dgsem_device_cache.D_d = cache.D.Read();
    dgsem_device_cache.Dhat_d = cache.Dhat.Read();
    dgsem_device_cache.Dhat2_d = cache.Dhat2.Read();
    dgsem_device_cache.nor_d = cache.face_normals.Read();
    dgsem_device_cache.fw_minus_d = cache.face_wt_minus.Read();
    dgsem_device_cache.fw_plus_d = cache.face_wt_plus.Read();

    // Updated every step by the compute device
    dgsem_device_cache.elWaveSpeed_d = cache.elWaveSpeed.Write();
    dgsem_device_cache.ifWaveSpeed_d = cache.ifWaveSpeed.Write();
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

// Set up and populate elJac, elMetric, D, Dhat, Dhat2
// Face normals, and weights
void DGSEMNonlinearForm::AssembleGeometricTerms()
{
  int nelem = cache.num_elements;
  int p = cache.p;
  int Np = cache.Np;
  int dim = cache.dim;
  int Np_x = Np;
  int Np_y = dim > 1 ? Np : 1;
  int Np_z = dim > 2 ? Np : 1;
  int neq = cache.num_equations;
  Mesh *mesh = fes->GetMesh();

  // Build integration rules
  int IntegrationOrder = 2 * Np_x - 3;
  cache.ir = &cache.GLIntRules.Get(mfem::Geometry::SEGMENT, IntegrationOrder);
  if (dim == 1)
    {
      cache.ir_face = &cache.GLIntRules.Get(mfem::Geometry::POINT, IntegrationOrder);
      cache.ir_vol = &cache.GLIntRules.Get(mfem::Geometry::SEGMENT, IntegrationOrder);
    }
  else if (dim == 2)
    {
      cache.ir_face = &cache.GLIntRules.Get(mfem::Geometry::SEGMENT, IntegrationOrder);
      cache.ir_vol = &cache.GLIntRules.Get(mfem::Geometry::SQUARE, IntegrationOrder);
    }
  else
    {
      cache.ir_face = &cache.GLIntRules.Get(mfem::Geometry::SQUARE, IntegrationOrder);
      cache.ir_vol = &cache.GLIntRules.Get(mfem::Geometry::CUBE, IntegrationOrder);
    }
  
  MFEM_ASSERT(cache.ir->GetNPoints() == Np_x, "");
  MFEM_ASSERT(cache.ir_vol->GetNPoints() == Np_x*Np_y*Np_z, "");

  // Populate element Jacobian determinant and metric terms
  cache.elJac.SetSize(Np_x*Np_y*Np_z*nelem);
  cache.elMetric.SetSize(dim*dim*Np_x*Np_y*Np_z*nelem);
  for (int i = 0; i < nelem; i++)
    {
      ElementTransformation *T = fes->GetElementTransformation(i);
      assert(T->ElementNo == i);
      AssembleElementVolumeGeometricTerms(*T);
    }

  // Set up derivative operators
  mfem::DenseMatrix D_T, Dhat_T, Dhat2_T;
  D_T.SetSize(Np_x);
  Dhat_T.SetSize(Np_x);
  Dhat2_T.SetSize(Np_x);
 
  Vector wBary(Np_x);
  wBary = 1.0;
  
  for (int i = 1; i < Np_x; i++)
    {
      for (int j = 0; j < i; j++)
        {
          wBary(j) *= (cache.ir->IntPoint(j).x - cache.ir->IntPoint(i).x);
          wBary(i) *= (cache.ir->IntPoint(i).x - cache.ir->IntPoint(j).x);
        }
    }
  
  wBary.Reciprocal();
  D_T = 0.0;
  for (int iL = 0; iL < Np_x; iL++)
    {
      for (int i = 0; i < Np_x; i++)
        {
          if (iL != i)
            {
              D_T(i, iL) = wBary(iL) / wBary(i) / (cache.ir->IntPoint(i).x - cache.ir->IntPoint(iL).x);
              D_T(i, i) -= D_T(i, iL);
            }
        }
    }
  
  Dhat_T = D_T;
  Dhat_T(0, 0) += 1.0 / cache.ir->IntPoint(0).weight;
  Dhat_T(Np - 1, Np - 1) -= 1.0 / cache.ir->IntPoint(Np - 1).weight;
  Dhat_T.Transpose();
  
  Dhat2_T = D_T;
  Dhat2_T *= 2.0;
  Dhat2_T(0, 0) += 1.0 / cache.ir->IntPoint(0).weight;
  Dhat2_T(Np - 1, Np - 1) -= 1.0 / cache.ir->IntPoint(Np - 1).weight;
  Dhat2_T.Transpose();
  D_T.Transpose();

  // Just copy D_T, Dhat_T, and Dhat2_T
  cache.D.SetSize(Np_x*Np_x);
  cache.Dhat.SetSize(Np_x*Np_x);
  cache.Dhat2.SetSize(Np_x*Np_x);
  std::memcpy(cache.D.GetData(),     D_T.Data(),     sizeof(real_t)*Np_x*Np_x);
  std::memcpy(cache.Dhat.GetData(),  Dhat_T.Data(),  sizeof(real_t)*Np_x*Np_x);
  std::memcpy(cache.Dhat2.GetData(), Dhat2_T.Data(), sizeof(real_t)*Np_x*Np_x);

  // Set up data for faces
  const int nfp = cache.ir_face->GetNPoints();
  cache.num_face_points = nfp;

  const int nfaces_restr = cache.restr_f->Height() / (nfp * neq * 2);
  cache.num_interior_faces = nfaces_restr;
  MFEM_VERIFY(nfaces_restr > 0, "nfaces_restr is 0");

  cache.face_normals.SetSize(nfaces_restr * nfp * dim);
  cache.face_wt_minus.SetSize(nfaces_restr * nfp);
  cache.face_wt_plus.SetSize(nfaces_restr * nfp);
  AssembleFaceGeomCacheInterior();

}

// Builds element-specific Jac/Metric and stuffs into cache.elJac, cache.elMetric
void DGSEMNonlinearForm::AssembleElementVolumeGeometricTerms(ElementTransformation &Tr)
{

  real_t *Jinv_h = cache.elJac.HostWrite();
  real_t *Met_h  = cache.elMetric.HostWrite();
  int dim = cache.dim;
  mfem::Vector metric1(dim);
  const int e = Tr.ElementNo;
  const int nq = cache.Np_x * cache.Np_y * cache.Np_z;

  for (int q = 0; q < nq; ++q)
    {
      const IntegrationPoint &ip = cache.ir_vol->IntPoint(q);
      Tr.SetIntPoint(&ip);
      const real_t J = Tr.Weight();
      Jinv_h[e*nq + q] = J;
      
      const mfem::DenseMatrix &adj = Tr.AdjugateJacobian();              
      for (int dir = 0; dir < dim; ++dir)
        {
          adj.GetRow(dir, metric1);  // metric1.Size() == dim
          
          for (int d = 0; d < dim; ++d)
            {
              const int idxM = (((e*nq + q)*dim + dir)*dim + d);
              Met_h[idxM] = metric1(d);
            }
        }
    }
}

// This is called from OUTSIDE by DGSEMOperator.cpp
// Because this call requires some setup to be done
// *after* instantiation : can't be called from
// constructor.
  void DGSEMNonlinearForm::CreateOperatorCache()
  {
    MFEM_VERIFY(fes, "fes must be set");
    Mesh *mesh = fes->GetMesh();
    int p = fes->GetFE(0)->GetOrder();
    int dim = mesh->SpaceDimension();
    int Np = p + 1;
    int Np_x = Np;
    int Np_y = dim > 1 ? Np : 1;
    int Np_z = dim > 2 ? Np : 1;
  
    cache.p = p;
    cache.Np = Np;
    cache.dim = dim;
    cache.Np_x = Np_x;
    cache.Np_y = Np_y;
    cache.Np_z = Np_z;

    MFEM_VERIFY(mesh, "mesh must be set");
    MFEM_VERIFY(dnfi[0] != nullptr, "Domain integrator must be set.");

    // ---- sizes / metadata ----------------------------------------------------
    const int ne = fes->GetNE();
    cache.num_elements = ne;
    // Attribute count = max attribute id (1-based in MFEM)
    cache.num_attr = mesh->attributes.Size() ? mesh->attributes.Max() : 0;
    cache.ndof_scalar_el = fes->GetFE(0)->GetDof();
    cache.num_equations = fes->GetVDim();
    cache.dim = mesh->SpaceDimension();

    // ---- 1) Build combined attribute marker exactly like Mult() --------------
    cache.attr_marker.SetSize(cache.num_attr);
    cache.attr_marker = 0;
    
    if (dnfi.Size() == 0 || cache.num_attr == 0)
      {
        // If no domain integrators, nothing to do; marker stays 0.
        // If "process all" is desired instead, set marker=1 here.
        cache.attr_marker = 1;
      }
    else
      {
        for (int k = 0; k < dnfi.Size(); k++)
          {
            if (dnfi_marker[k] == nullptr)
              {
                cache.attr_marker = 1; // process all attrs
                break;
              }

            const Array<int> &marker = *dnfi_marker[k];
            MFEM_ASSERT(marker.Size() == cache.attr_marker.Size(),
                        "invalid marker for domain integrator #" << k);
            
            for (int i = 0; i < cache.attr_marker.Size(); i++)
              {
                cache.attr_marker[i] |= marker[i];
              }
          }
      }

    // ---- 1b) Cache per-integrator marker (single dnfi assumption) -------------
    cache.dnfi_marker.SetSize(cache.num_attr);
    cache.dnfi_marker = 0;
    
    if (dnfi.Size() == 0 || cache.num_attr == 0)
      {
        cache.dnfi_marker = 1;
      }
    else
      {
        MFEM_VERIFY(dnfi.Size() == 1, "expected exactly one dnfi integrator");
        
        if (dnfi_marker[0] == nullptr)
          {
            cache.dnfi_marker = 1; // applies to all attrs
          }
        else
          {
            const mfem::Array<int> &m0 = *dnfi_marker[0];
            MFEM_ASSERT(m0.Size() == cache.num_attr, "invalid dnfi_marker[0] size");
            
            for (int a = 0; a < cache.num_attr; ++a)
              {
                cache.dnfi_marker[a] = m0[a];
              }
          }
      }
    
    // ---- 2) Per-element attribute id array -----------------------------------
    cache.elem_attr.SetSize(ne);
    for (int e = 0; e < ne; ++e)
      {
        const int attr = mesh->GetAttribute(e); // 1-based
        cache.elem_attr[e] = attr;
      }

    // Optional host-side sanity check (cheap, catches bad markers early):
    if (cache.num_attr > 0)
      {
        for (int e = 0; e < ne; ++e)
          {
            const int a = cache.elem_attr[e];
            MFEM_VERIFY(a >= 1 && a <= cache.num_attr,
                        "element attribute out of range: attr=" << a
                        << " num_attr=" << cache.num_attr);
          }
      }

    cache.elem_attr.UseDevice();
    cache.attr_marker.UseDevice();
    cache.dnfi_marker.UseDevice();
    cache.elem_attr.Read();
    cache.attr_marker.Read();
    cache.dnfi_marker.Read();

    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    cache.restr_v = fes->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
    cache.restr_f = pfes->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                             mfem::FaceType::Interior,
                                             mfem::L2FaceValues::DoubleValued);
    cache.elWaveSpeed.SetSize(ne);
    cache.elWaveSpeed = 0.0;
    cache.elWaveSpeed.UseDevice();
    cache.elWaveSpeed.Read();

    AssembleGeometricTerms();

    cache.elJac.UseDevice();
    cache.elMetric.UseDevice();
    cache.D.UseDevice();
    cache.Dhat.UseDevice();
    cache.Dhat2.UseDevice();
    cache.elJac.Read();
    cache.elMetric.Read();
    cache.D.Read();
    cache.Dhat.Read();
    cache.Dhat2.Read();

    cache.face_normals.UseDevice();
    cache.face_wt_minus.UseDevice();
    cache.face_wt_plus.UseDevice();
    cache.face_normals.Read();
    cache.face_wt_minus.Read();
    cache.face_wt_plus.Read();

    cache.ifWaveSpeed.SetSize(cache.num_interior_faces);
    cache.ifWaveSpeed = 0.0;
    cache.ifWaveSpeed.UseDevice();
    cache.ifWaveSpeed.Read();

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

// Original MULT routine: kept around for reference until refactor done
void DGSEMNonlinearForm::MultOG(const Vector &u, Vector &dudt) const
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

// Intermediate version of MULT starting to evolved toward device use
void DGSEMNonlinearForm::Mult(const Vector &u, Vector &dudt) const
{
    const Vector &pu = Prolongate(u);
    
    if (P)
    {
        aux2.SetSize(P->Height());
    }
    Vector &pdudt = P ? aux2 : dudt;
    pdudt = 0.0;

    Array<int> vdofs;
    Vector el_u, el_dudt;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    const int *attr_marker = cache.attr_marker.Read();
    const int *elem_attr = cache.elem_attr.Read();
    const int *dnfi_marker = cache.dnfi_marker.Read();

    if (dnfi.Size())
      {
        for (int i = 0; i < fes->GetNE(); i++)
          {
            const int attr = elem_attr[i];
            if (attr_marker[attr-1] == 0) { continue; }
            if (dnfi_marker[attr-1] == 0) { continue; }
            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            // for (int k = 0; k < dnfi.Size(); k++)
            //{
            //  if(dnfi_marker[attr-1] == 0) { continue; }
            dnfi[0]->AssembleElementVector(*fe, *T, el_u, el_dudt);
            pdudt.AddElementVector(vdofs, el_dudt);
            // }
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

// Host version - but uses some cached operator stuff, just a test
real_t DGSEMNonlinearForm::MultInviscidVolumeHost(const Vector &pu, Vector &pdudt) const
{

    Array<int> vdofs;
    Vector el_u, el_dudt;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();

    const int *attr_marker = cache.attr_marker.Read();
    const int *elem_attr = cache.elem_attr.Read();
    const int *dnfi_marker = cache.dnfi_marker.Read();
    real_t *ws_d = cache.elWaveSpeed.ReadWrite();
    real_t max_char_speed = 0.0;

    if (dnfi.Size())
      {
        for (int i = 0; i < fes->GetNE(); i++)
          {
            const int attr = elem_attr[i];
            if (attr_marker[attr-1] == 0) {
              ws_d[i] = 0.0;
              continue;
            }
            // if (dnfi_marker[attr-1] == 0) { continue; }
            fe = fes->GetFE(i);
            fes->GetElementVDofs(i, vdofs);
            T = fes->GetElementTransformation(i);
            pu.GetSubVector(vdofs, el_u);
            // for (int k = 0; k < dnfi.Size(); k++)
            //{
            //  if(dnfi_marker[attr-1] == 0) { continue; }
            ws_d[i] = dnfi[0]->AssembleElementVectorHost(*fe, *T, el_u, el_dudt);
            pdudt.AddElementVector(vdofs, el_dudt);
            // }
        }

        const real_t *ws = cache.elWaveSpeed.Read();
        for(int e = 0;e < cache.num_elements;e++)
          {
            max_char_speed = std::max(max_char_speed, ws[e]);
          }
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
real_t DGSEMNonlinearForm::MultVolumeInviscidDevice(const Vector &pu, Vector &pdudt) const
{
  // ScopedTimer timer("MultVolumeInviscidDevice");

  // This block is executed by the host
  mfem::Vector Ue(cache.restr_v->Height());
  mfem::Vector dUe(cache.restr_v->Height());
  Ue.UseDevice();
  dUe.UseDevice();

  cache.restr_v->Mult(pu, Ue);
  
  // If you want dUe zeroed before accumulation, do it explicitly on device:
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
  // ChatGPT advised to grab a new write location every step
  // real_t *ws_d = cache.elWaveSpeed.Write();

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

    const real_t cs_el = \
      DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                   jac_el, metric_el, du_el);
    ws_d[e] = cs_el;
  });
  //  mfem::Device::Synchronize();

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  //  - Scatter RHS back to storage
  const real_t *ws = cache.elWaveSpeed.HostRead();
  real_t max_char_speed = 0.0;
  for(int e = 0;e < cache.num_elements;e++)
    {
      max_char_speed = std::max(max_char_speed, ws[e]);
    }
  cache.restr_v->AddMultTranspose(dUe, pdudt);
  //  mfem::Device::Synchronize();
  return max_char_speed;
}

void DGSEMNonlinearForm::MultInteriorFacesInviscidHost(const Vector &pu, Vector &pdudt) const
{
  //  HostFaceAssemblyDebug(pu, pdudt);
  Array<int> vdofs;
  Vector el_u, el_dudt;
  const FiniteElement *fe;
  ElementTransformation *T;
  Mesh *mesh = fes->GetMesh();
  //  std::ostringstream Ostr;
  const int Np = cache.Np;

  if (fnfi.Size())
    {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;
      auto &int_faces = mesh->GetFaceIndices(mfem::FaceType::Interior);
      int nfaces_int = int_faces.Size();
      for (int iface = 0; iface < nfaces_int; iface++)
        {
          const int face_id = int_faces[iface];
          MFEM_VERIFY(iface==face_id, "Faces dont line up");
          tr = mesh->GetInteriorFaceTransformations(face_id);
          MFEM_VERIFY(tr, "Expected transformation");
          if (tr != NULL)
            {
              fes->GetElementVDofs(tr->Elem1No, vdofs);
              fes->GetElementVDofs(tr->Elem2No, vdofs2);
              vdofs.Append (vdofs2);              
              pu.GetSubVector(vdofs, el_u);
              fe1 = fes->GetFE(tr->Elem1No);
              fe2 = fes->GetFE(tr->Elem2No);
              // INSERT PRINT TEST HERE              
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
                  fnfi[k]->AssembleFaceVectorInviscid(*fe1, *fe2, *tr, el_u, el_dudt);
                  aux2.AddElementVector(vdofs1, el_dudt.GetData());
                }
            }
        }
    }
}

real_t DGSEMNonlinearForm::MultInteriorFacesInviscidDevice(const Vector &pu, Vector &pdudt) const
{
  // CheckFaceOrderings(pu);
  //  ScopedTimer timer("MultInteriorFacesInviscidDevice");
  auto dc = device_cache;
  const int dim = dc.dim; // pfes->GetMesh()->Dimension();
  const int neq = dc.num_equations; // pfes->GetVDim();
  const int nfp = dc.num_face_points; // cache.num_face_points; // ir_face->GetNPoints();
  const int nfaces = cache.restr_f->Height() / (nfp * neq * 2); // (+/-)
  const int face_stride = 2 * nfp * neq;
  const int side_stride = nfp * neq;
  const int face_size = 2*nfp*neq;
  const int norm_size = nfp*dim;
  
  // TODO: Move these to where the caches are created and validated
  // MFEM_VERIFY(nfaces == cache.num_interior_faces, "restriction faces != cached interior faces");
  // MFEM_VERIFY(cache.face_normals.Size() == nfaces*nfp*dim, "normals size mismatch");
  // MFEM_VERIFY(cache.face_wt_minus.Size() == nfaces*nfp, "w_minus size mismatch");
  // MFEM_VERIFY(cache.face_wt_plus.Size()  == nfaces*nfp, "w_plus size mismatch");

  mfem::Vector u_faces(cache.restr_f->Height());
  mfem::Vector rhs_faces(cache.restr_f->Height());
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
  //  mfem::Device::Synchronize();
  cache.restr_f->Mult(pu, u_faces);

  const real_t *u_d = u_faces.Read();
  real_t *rhs_d = rhs_faces.Write();

  const real_t *nor_d   = dc.nor_d;      // cache.face_normals.Read();   // size nfaces*nfp*dim
  const real_t *inv1_d  = dc.fw_minus_d; // .Read();  // size nfaces*nfp
  const real_t *inv2_d  = dc.fw_plus_d;  // .Read();   // size nfaces*nfp

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
  //  mfem::Device::Synchronize();
  cache.restr_f->MultTranspose(rhs_faces, faces_dudt);
  pdudt += faces_dudt;

  // Finish up on the host:
  //  - Reduce for rank-local max_char_speed
  const real_t *ws = cache.ifWaveSpeed.HostRead();
  real_t max_char_speed_facial = 0.0;
  for(int f = 0;f < cache.num_interior_faces;f++)
    {
      max_char_speed_facial = std::max(max_char_speed_facial, ws[f]);
    }

  return max_char_speed_facial;
}

// Top level MULT for inviscid cases, called from DGSEMOperator
real_t DGSEMNonlinearForm::MultInviscid(const Vector &u, Vector &dudt) const
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
    // max_char_speed = MultInviscidVolumeHost2(pu, pdudt);
    max_char_speed = MultVolumeInviscidDevice(pu, pdudt);
    // std::cout << "Volume wavespeed: " << max_char_speed << std::endl;

    // Testing this during development:
    real_t max_char_speed_facial = 0.0;
    max_char_speed_facial = MultInteriorFacesInviscidDevice(pu, pdudt);
    // std::cout << "Facial wavespeed: " << max_char_speed_facial << std::endl;
    // MultInteriorFacesInviscidHost(pu, pdudt);
    max_char_speed = std::max(max_char_speed, max_char_speed_facial);

    Array<int> vdofs;
    Vector el_u, el_dudt;
    const FiniteElement *fe;
    ElementTransformation *T;
    Mesh *mesh = fes->GetMesh();


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

