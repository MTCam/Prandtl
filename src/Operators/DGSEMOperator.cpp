#include "DGSEMOperator.hpp"

namespace Prandtl
{
  DGSEMOperator::DGSEMOperator(std::shared_ptr<ParFiniteElementSpace> vfes_,
                               std::shared_ptr<ParFiniteElementSpace> fes0_,
                               std::shared_ptr<ParMesh> pmesh_,
                               std::shared_ptr<ParGridFunction> eta_,
                               std::shared_ptr<ParGridFunction> alpha_,
                               std::vector<std::shared_ptr<ParGridFunction> > &grad_u_,
                               std::unique_ptr<DGSEMIntegrator> integrator_,
                               std::unique_ptr<Indicator> indicator_,
                               const IdealGasModel &gasModel_,
                               std::shared_ptr<ParGridFunction> r_gf_,
                               const real_t alpha_max, const real_t alpha_min)
  : TimeDependentOperator(vfes_->GetTrueVSize()),
    vfes(vfes_), fes0(fes0_), pmesh(pmesh_),
    eta(eta_), alpha(alpha_), grad_u(grad_u_),
    integrator(std::move(integrator_)), indicator(std::move(indicator_)),
    gasModel(gasModel_),
    num_equations(vfes->GetVDim()), dim(pmesh->SpaceDimension()),
    order(vfes->GetElementOrder(0)), num_elements(pmesh->GetNE()),
    Ndofs(vfes->GetFE(0)->GetDof()),
    modalThreshold(0.5 * std::pow(10.0, -1.8 * std::pow(order, 0.25))),
    r_gf(r_gf_), alpha_max(alpha_max), alpha_min(alpha_min),
    num_dofs_scalar(vfes_->GetTrueVSize()/vfes_->GetVDim())
#ifdef AXISYMMETRIC
  , U(vfes->GetTrueVSize())
#endif
  {
    nonlinearForm.reset(new DGSEMNonlinearForm(vfes.get()));
    
    nonlinearForm->AddDomainIntegrator(integrator.get());
    nonlinearForm->AddInteriorFaceIntegrator(integrator.get());
    
    std::vector<BdrFaceIntegrator*>::iterator it1 = bfnfi.begin();
    std::vector<Array<int>>::iterator it2 = bdr_marker.begin();

    for (; it1 != bfnfi.end() && it2 != bdr_marker.end(); ++it1, ++it2)
    {
        nonlinearForm->AddBdrFaceIntegrator(*it1, *it2);
    }
    nonlinearForm->UseExternalIntegrators();

#ifdef PARABOLIC
    global_entropy.SetSize(vfes->GetVSize());
#endif
    CreateOperatorCache();
    OperatorCacheToDeviceCache();
    nonlinearForm->SetOperatorCache(&cache);
    nonlinearForm->SetDeviceCache(device_cache);
  }

  DGSEMOperator::~DGSEMOperator()
  {
    for (auto ptr : bfnfi)
      {
        delete ptr;
      }
  }

  void DGSEMOperator::CreateOperatorCache()
  {
    
    const int p = vfes->GetFE(0)->GetOrder();
    const int dim = pmesh->SpaceDimension();
    const int ndof_scalar_el = vfes->GetFE(0)->GetDof();
    const int neq = vfes->GetVDim();
    const int Np = p + 1;
    const int Np_x = Np;
    const int Np_y = dim > 1 ? Np : 1;
    const int Np_z = dim > 2 ? Np : 1;
    const int nelem = vfes->GetNE();
    const int nattr = pmesh->attributes.Size() ? pmesh->attributes.Max() : 0;
  
    cache.p = p;
    cache.Np = Np;
    cache.dim = dim;
    cache.Np_x = Np_x;
    cache.Np_y = Np_y;
    cache.Np_z = Np_z;
    cache.ndof_scalar_el = ndof_scalar_el;
    cache.num_equations = neq;
    cache.num_elements = nelem;
    cache.restr_v = vfes->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
    cache.restr_f = vfes->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                             mfem::FaceType::Interior,
                                             mfem::L2FaceValues::DoubleValued);


    // Attribute count = max attribute id (1-based in MFEM)
    cache.num_attr = nattr;
    cache.attr_marker.SetSize(nattr);
    cache.attr_marker = 1;

    // if (cache.num_attr == 0)
    //   {
    //     // If no domain integrators, nothing to do; marker stays 0.
    //     // If "process all" is desired instead, set marker=1 here.
    //     cache.attr_marker = 1;
    //   }
    // else
    //   {
    //     for (int k = 0; k < cache.volume_integrators.Size(); k++)
    //       {
    //         if (cache.volume_element_markers[k] == nullptr)
    //           {
    //             cache.attr_marker = 1; // process all attrs
    //             break;
    //           }

    //         const Array<int> &marker = *cache.volume_element_markers[k];
    //         MFEM_ASSERT(marker.Size() == cache.attr_marker.Size(),
    //                     "invalid marker for domain integrator #" << k);
            
    //         for (int i = 0; i < cache.attr_marker.Size(); i++)
    //           {
    //             cache.attr_marker[i] |= marker[i];
    //           }
    //       }
    //   }

    // ---- 1b) Cache per-integrator marker (single dnfi assumption) -------------
    cache.dnfi_marker.SetSize(nattr);
    cache.dnfi_marker = 1;
    
    // if (cache.volume_integrators.Size() == 0 || cache.num_attr == 0)
    //   {
    //     cache.dnfi_marker = 1;
    //   }
    // else
    //   {
    //     MFEM_VERIFY(cache.volume_integrators.Size() == 1, "expected exactly one dnfi integrator");
        
    //     if (cache.volume_element_markers[0] == nullptr)
    //       {
    //         cache.dnfi_marker = 1; // applies to all attrs
    //       }
    //     else
    //       {
    //         const mfem::Array<int> &m0 = *cache.volume_element_markers[0];
    //         MFEM_ASSERT(m0.Size() == cache.num_attr, "invalid dnfi_marker[0] size");
            
    //         for (int a = 0; a < cache.num_attr; ++a)
    //           {
    //             cache.dnfi_marker[a] = m0[a];
    //           }
    //       }
    //   }
    
    // ---- 2) Per-element attribute id array -----------------------------------
    cache.elem_attr.SetSize(nelem);
    for (int e = 0; e < nelem; ++e)
      {
        const int attr = pmesh->GetAttribute(e); // 1-based
        cache.elem_attr[e] = attr;
      }

    // Optional host-side sanity check (cheap, catches bad markers early):
    if (cache.num_attr > 0)
      {
        for (int e = 0; e < nelem; ++e)
          {
            const int a = cache.elem_attr[e];
            MFEM_VERIFY(a >= 1 && a <= cache.num_attr,
                        "element attribute out of range: attr=" << a
                        << " num_attr=" << cache.num_attr);
          }
      }

    bool ud = cache.elem_attr.UseDevice();
    //    MFEM_VERIFY(ud, "Device is off");
    cache.attr_marker.UseDevice();
    cache.dnfi_marker.UseDevice();
    cache.elem_attr.Read();
    cache.attr_marker.Read();
    cache.dnfi_marker.Read();

    cache.elWaveSpeed.SetSize(nelem);
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

  // Set up and populate elJac, elMetric, D, Dhat, Dhat2
  // Face normals, and weights
  void DGSEMOperator::AssembleGeometricTerms()
  {
    int nelem = cache.num_elements;
    int p = cache.p;
    int Np = cache.Np;
    int dim = cache.dim;
    int Np_x = Np;
    int Np_y = dim > 1 ? Np : 1;
    int Np_z = dim > 2 ? Np : 1;
    int neq = cache.num_equations;
    
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
        ElementTransformation *T = vfes->GetElementTransformation(i);
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
    AssembleInteriorFaceGeomCache(); 
  }

  // Builds element-specific Jac/Metric and stuffs into cache.elJac, cache.elMetric
  void DGSEMOperator::AssembleElementVolumeGeometricTerms(ElementTransformation &Tr)
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

  void DGSEMOperator::AssembleInteriorFaceGeomCache()
  {
    // auto *mesh = fes->GetMesh();
    // auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    // auto *pfes = vfes;// dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    cache.fqs_int = new mfem::FaceQuadratureSpace(*pmesh, *cache.ir_face, mfem::FaceType::Interior);
    // MFEM_VERIFY(vfes, "need ParFiniteElementSpace");
    
    const int dim = cache.dim;
    const int neq = cache.num_equations;
    const int nfp = cache.num_face_points; // cache.ir_face->GetNPoints();

    auto &int_faces = pmesh->GetFaceIndices(mfem::FaceType::Interior);
    BuildFaceLists();

    const int nfaces = int_faces.Size();

    // inv_map_all[face_slot*nfp + fp_perm] = fp_restr
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
      // if(fslot == 0){
      //   std::cout << "Normal = " << nor(0) << "," << nor(1)
      //             << std::endl;
      // }
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
          std::cout << "Shared face (slot/face): (" << fslot << "/" << face_id << ")"
                    << std::endl;
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
              // std::cout << "J1/J2 = " << J1 << "/" << J2 << std::endl;
              store(fslot, fp_restr, nor, fac1/(w0*J1), fac2/(w0*J2));
            }
        } else { // local internal face
          std::cout << "Internal face (slot/face): (" << fslot << "/" << face_id << ")"
                    << std::endl;
          auto *tr = pmesh->GetInteriorFaceTransformations(face_id);
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

  void DGSEMOperator::OperatorCacheToDeviceCache()
  {
    device_cache.gas = gasModel;
    device_cache.ndof_scalar_el = cache.ndof_scalar_el;
    device_cache.num_attr = cache.num_attr;
    device_cache.attr_marker_d = cache.attr_marker.Read();
    device_cache.elem_attr_d = cache.elem_attr.Read();
    device_cache.elWaveSpeed_d = cache.elWaveSpeed.ReadWrite();
    device_cache.ifWaveSpeed_d = cache.ifWaveSpeed.ReadWrite();
    device_cache.num_face_points = cache.num_face_points;
    device_cache.p = cache.p;
    device_cache.dim = cache.dim;
    device_cache.Np = cache.Np;
    device_cache.Np_x = cache.Np_x;
    device_cache.Np_y = cache.Np_y;
    device_cache.Np_z = cache.Np_z;
    device_cache.num_elements = cache.num_elements;
    device_cache.num_equations = cache.num_equations;
    device_cache.elJac_d = cache.elJac.Read();
    device_cache.elMetric_d = cache.elMetric.Read();
    device_cache.D_d = cache.D.Read();
    device_cache.Dhat_d = cache.Dhat.Read();
    device_cache.Dhat2_d = cache.Dhat2.Read();
  }
  
  // Call once after:
  //  - fes is finalized
  //  - dnfi/dnfi_marker are set
  //  - DGSEMIntegrator has built elJac/elMetric, and Dcol
  // and before entering the time loop.
  // void DGSEMOperator::AssembleDeviceCache()
  // {
  //   // Copy the POD gasmodel outright 
  //   device_cache.gas = gasModel;
  //   // Get the integrator's device-ready cache data
  //   // Populates J, M, D, WS
  //   // GetDeviceCache();
    
  // }

void DGSEMOperator::ComputeBlendingCoefficient(const Vector &x) const
{
    indicator->CheckSmoothness(x);
    for (int el = 0; el < num_elements; el++)
    {
        fes0->GetElementDofs(el, ind_indx);
        eta->GetSubVector(ind_indx, ind_dof);
        alpha_dof = 1.0 / (1.0 + std::exp(-sharpness_fac * (ind_dof(0) - modalThreshold) / modalThreshold));
        if (alpha_dof < alpha_min)
        {
            alpha_dof = 0.0;
        }
        else if (alpha_dof > (1.0 - alpha_min))
        {
            alpha_dof = 1.0;
        }
        alpha_dof = std::min(alpha_dof, alpha_max);
        alpha->SetSubVector(ind_indx, alpha_dof);
    }
    
}

void DGSEMOperator::ComputeGlobalEntropyVector(const Vector &u, Vector &global_entropy) const
{
    DenseMatrix ent_mat(Ndofs, num_equations);
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);
        Conserv2Entropy(gasModel, vdof_mat, ent_mat);
        global_entropy.SetSubVector(vdof_indices, ent_mat.GetData());
    }
}

void DGSEMOperator::ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx) const
{
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);

        dudx.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat);
        dudx.SetSubVector(vdof_indices, grad_mat.GetData());
    }
}

void DGSEMOperator::ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy) const
{
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);

        dudx.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat1(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat1);
        dudx.SetSubVector(vdof_indices, grad_mat1.GetData());

        dudy.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat2(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat2);
        dudy.SetSubVector(vdof_indices, grad_mat2.GetData());    
        
    }
}

void DGSEMOperator::ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const
{
    for (int el = 0; el < num_elements; el++)
    {
        ElementTransformation *Tr = vfes->GetElementTransformation(el);
        vfes->GetElementVDofs(el, vdof_indices);
        u.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), Ndofs, num_equations);

        dudx.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat1(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat1);
        dudx.SetSubVector(vdof_indices, grad_mat1.GetData());

        dudy.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat2(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat2);
        dudy.SetSubVector(vdof_indices, grad_mat2.GetData());

        dudz.GetSubVector(vdof_indices, grad_vdofs);
        DenseMatrix grad_mat3(grad_vdofs.GetData(), Ndofs, num_equations);
        EntropyGrad2PrimGrad(gasModel, vdof_mat, grad_mat3);
        dudz.SetSubVector(vdof_indices, grad_mat3.GetData());      
    }
}

#ifdef AXISYMMETRIC

    void DGSEMOperator::RecoverStateFromWeighted(const Vector &rU, Vector &U) const
    {
        U.SetSize(rU.Size());

        const real_t tiny = 1e-14;
        const real_t cap_mult = 10.0;
        const real_t z_tol = 1e-12;
        const real_t tiny_detJ = 1e-12;
        const real_t theta_lim = 1.5;
        const real_t rho_floor = rho_floor_abs;
        const real_t p_floor = p_floor_abs;
        long long calls = 0;
        
        enum class AxisReconMode { highOrder_shape, lowOrder_ray1, lowOrder_ray2, lowOrder_copy };
        
        // helper functions
        auto sign = [](real_t a) -> real_t { return (a > 0) - (a < 0); };

        auto minmod = [&](real_t a, real_t b) -> real_t
        {
            if (a * b <= 0.0) { return 0.0; }
            return sign(a) * std::min(std::abs(a), std::abs(b));
        };

        auto clamp = [&](real_t x, real_t m) -> real_t
        {
            if (!isfinite(x)) { return 0.0; }
            const real_t ax = std::abs(x);
            return (ax <= m) ? x : sign(x) * m;
        };

        auto is_dof_on_axis = [&](int dof_id) -> bool 
        {
            const int* beg = axis_idx.GetData();
            const int* end = beg + axis_idx.Size();
            return std::binary_search(beg, end, dof_id);
        };

        Array<int> vdofs;

        for (int e = 0; e < vfes->GetNE(); e++)
        {
            vfes->GetElementVDofs(e, vdofs);

            bool troubled = false;

            if (alpha && fes0)
            {
                Array<int> ind;
                fes0->GetElementDofs(e, ind);
                Vector a_loc;
                alpha->GetSubVector(ind, a_loc);
                troubled = (a_loc.Size() > 0) ? (a_loc(0) > 0.5 * alpha_max) : false;
            }

            Vector rU_e, r_e;
            rU.GetSubVector(vdofs, rU_e);
            MFEM_ASSERT(r_gf != nullptr, "r_gf is null");
            r_gf->GetSubVector(vdofs, r_e);

            Vector U_e(rU_e.Size());

            const DenseMatrix rU_e_mat(rU_e.GetData(), Ndofs, num_equations);
            DenseMatrix U_e_mat(U_e.GetData(), Ndofs, num_equations);
            const Vector r_e_vec(r_e.GetData(), Ndofs);

            const FiniteElement& fe = *vfes->GetFE(e);
            ElementTransformation& Tr = *vfes->GetElementTransformation(e);
            const IntegrationRule& nodes = fe.GetNodes();

            troubled = troubled || low_order_axis;

            for (int ld = 0; ld < Ndofs; ld++)
            {
                const real_t r = r_e_vec(ld);
                const int true_dof_id = vdofs[ld];
                const bool is_axis_node = is_dof_on_axis(true_dof_id);

                // true axis nodes and near axis nodes

                if (!is_axis_node && r > 0.0) // off axis nodes
                {
                    for (int eq = 0; eq < num_equations; eq++)
                    {
                        U_e_mat(ld, eq) = rU_e_mat(ld, eq) / r;
                    }
                    continue;
                }

                // on axis nodes
                calls++;
                AxisReconMode mode = AxisReconMode::highOrder_shape;

                // HIGH ORDER construction of U(0) = d/dr(rU)|(r=0)
                // df/dr = (J^-T * grad_ref f)_r
                const IntegrationPoint &ip = nodes.IntPoint(ld);
                Tr.SetIntPoint(&ip);

                DenseMatrix dshape(Ndofs, dim);
                fe.CalcDShape(ip, dshape);

                Vector X_axis(dim);
                Tr.Transform(ip, X_axis);
                const real_t z_axis = X_axis(0);

                Vector rrho_e(Ndofs), rmz_e(Ndofs), rE_e(Ndofs);

                for (int i = 0; i < Ndofs; i++)
                {
                    rrho_e(i) = rU_e_mat(i, 0);
                    rmz_e(i)  = rU_e_mat(i, 1);
                    rE_e(i)   = rU_e_mat(i, 3);
                }

                // derivative function with fallbacks
                auto d_dr = [&](const Vector& f_e)-> real_t
                {
                    real_t d_dxi = 0.0;
                    real_t d_deta = 0.0;
                    for (int i = 0; i < Ndofs; i++)
                    {
                        const real_t fa = f_e(i);
                        d_dxi += fa * dshape(i, 0);
                        d_deta += fa * dshape(i, 1);
                    }

                    const DenseMatrix &Jinv = Tr.InverseJacobian();
                    DenseMatrix JinvT(Jinv);
                    JinvT.Transpose();

                    Vector grad_ref(dim);
                    grad_ref = 0.0;
                    grad_ref(0) = d_dxi;
                    grad_ref(1) = d_deta;

                    Vector grad_phys(dim);
                    JinvT.Mult(grad_ref, grad_phys);
                    const real_t detJ = Tr.Weight();

                    const real_t df_dr_geom = grad_phys(1);

                    // same-z nearest off axis candidates
                    int j1 = -1, j2 = -1;
                    real_t r1 = infinity(), r2 = infinity();
                    
                    for (int j = 0; j < Ndofs; j++)
                    {
                        const real_t rj = r_e_vec(j);
                        if (rj <= 0.0) { continue; }
                        const IntegrationPoint &ipj = nodes.IntPoint(j);
                        Tr.SetIntPoint(&ipj);
                        Vector Xj(dim);
                        Tr.Transform(ipj, Xj);
                        const real_t zj = Xj(0);
                        if (std::abs(zj - z_axis) <= z_tol)
                        {
                            if (rj < r1)
                            {
                                r2 = r1;
                                j2 = j1;
                                r1 = rj;
                                j1 = j;
                            }
                            else if (rj < r2)
                            {
                                r2 = rj;
                                j2 = j;
                            }
                        }
                    }
                    // if no same z, fallback to globally nearest two off-axis
                    if (j1 < 0)
                    {
                        for (int j = 0; j < Ndofs; j++)
                        {
                            const real_t rj = r_e_vec(j);
                            if (rj > 0.0 && rj < r1)
                            {
                                r2 = r1;
                                j2 = j1;
                                r1 = rj;
                                j1 = j;
                            }
                            else if (rj > 0.0 && rj < r2)
                            {
                                r2 = rj;
                                j2 = j;
                            }
                        }
                    }

                    // cap large gradients
                    const bool have_cap = (j1 >= 0);
                    const real_t f_ref = have_cap ? std::abs(f_e(j1)) : 0.0;
                    const bool ref_too_small = f_ref < 1e-12;
                    const real_t cap = (have_cap && !ref_too_small) ?  (cap_mult * f_ref / std::max(r1, tiny)) : infinity();
                    const bool geom_ok = isfinite(df_dr_geom) && isfinite(detJ) && (std::abs(detJ) > tiny_detJ) && (std::abs(df_dr_geom) <= cap) && !troubled;

                    if (geom_ok)
                    {
                        mode = AxisReconMode::highOrder_shape;
                        return df_dr_geom;
                    }

                    // LOW ORDER fallbacks
                    // 1st order fallback: 2 points least squares along ray + limiter
                    if (j1 >= 0 && j2 >= 0 && isfinite(r1) && isfinite(r2) && r1 > 0.0 && r2 > 0.0)
                    {
                        const real_t f1 = f_e(j1), f2 = f_e(j2);
                        // fitting through the origin: f = mr -> m = (r1 f1 + r2 f2)/(r1^2 + r2^2)
                        real_t m = (r1*f1 + r2*f2) / (std::max(r1*r1 + r2*r2, tiny));
                        const real_t s1 = f1/r1;
                        const real_t s2 = f2/r2;
                        real_t m_limited = minmod(m, minmod(theta_lim * s1, theta_lim * s2));
                        m_limited = clamp(m_limited, cap);

                        mode = AxisReconMode::lowOrder_ray2;
                        return m_limited;
                    }

                    // 1st order fallback: 1 point + limiter
                    if (j1 >= 0 && isfinite(r1) && r1 > 0.0)
                    {
                        const real_t s = f_e(j1)/r1;
                        mode = AxisReconMode::lowOrder_ray1;
                        return clamp(s, cap);
                    }

                    // 0th order trigger
                    mode = AxisReconMode::lowOrder_copy;
                    return 0.0;
                };

                    real_t rho_axis = d_dr(rrho_e);
                    real_t mz_axis  = d_dr(rmz_e);
                    real_t E_axis   = d_dr(rE_e);

                    if (!isfinite(rho_axis) || rho_axis < rho_floor || !isfinite(E_axis))
                    {
                        // find nearest off-axis neighbor
                        real_t r_neighbor = -1.0;
                        int neighbor_ld = -1;

                        for (int nld = 0; nld < Ndofs; nld++)
                        {
                            const real_t rn = r_e_vec(nld);
                            if ( rn > 0.0 && (neighbor_ld == -1 || rn < r_neighbor)) 
                            {
                                r_neighbor = rn;
                                neighbor_ld = nld;
                            }
                        }
                        // if no same z, fallback to globally nearest off-axis
                        if (neighbor_ld != -1)
                        {
                            for (int eq = 0; eq < num_equations; eq++)
                            {
                               U_e_mat(ld, eq) = rU_e_mat(neighbor_ld, eq) / r_neighbor;
                            }
                            U_e_mat(ld, 2) = 0.0;
                            mode = AxisReconMode::lowOrder_copy;
                        }
                        
                        else
                        {
                            // abort
                            MFEM_ABORT("All axis reconstruction fallbacks failed for a node");
                            break;
                        }
                    }
                    else
                    {
                        // high/low order result
                        U_e_mat(ld, 0) = rho_axis;
                        U_e_mat(ld, 1) = mz_axis;
                        U_e_mat(ld, 2) = 0.0; // rho ur = 0
                        U_e_mat(ld, 3) = E_axis;
                    }

                    switch (mode)
                        {
                            case AxisReconMode::highOrder_shape: highOrder_shape_accum++;
                            break;
                            case AxisReconMode::lowOrder_ray1: lowOrder_ray1_accum++;  
                            break;  
                            case AxisReconMode::lowOrder_ray2: lowOrder_ray2_accum++;  
                            break;  
                            case AxisReconMode::lowOrder_copy: lowOrder_copy_accum++;  
                            break;  
                        }
            }   
                        
            U.SetSubVector(vdofs, U_e);

        }
        calls_accum += calls;

    }

    DGSEMOperator::AxisReconStats DGSEMOperator::GetAxisReconStats(bool global) const
    {
        AxisReconStats s {calls_accum, highOrder_shape_accum , lowOrder_ray2_accum, lowOrder_ray1_accum, lowOrder_copy_accum};
        if (!global) return s;

        long long in[5] = { s.calls, s.highOrder_shape, s.lowOrder_ray2, s.lowOrder_ray1, s.lowOrder_copy }, out[5] = {0, 0, 0, 0, 0};
        MPI_Allreduce(in, out, 5, MPI_LONG_LONG, MPI_SUM, pmesh->GetComm());
        return AxisReconStats{ out[0], out[1], out[2], out[3], out[4]};
    }


    void DGSEMOperator::BuildAxisIndexFromMarker()
    {
        axis_idx.SetSize(0);
        if (axis_marker.Size() == 0) { return; }

        ParMesh &pm = *pmesh;
        FiniteElementSpace &fes_scalar = *vfes;

        Array<int> el_dofs;
        const int nbe = pm.GetNBE();

        for (int be = 0; be < nbe; be++)
        {
            const int a = pm.GetBdrAttribute(be);
            if (a <= 0 || a > axis_marker.Size() || axis_marker[a-1] == 0) { continue; }
            
            int el = -1, lf = -1;
            pm.GetBdrElementAdjacentElement(be, el, lf);
            MFEM_ASSERT(el >= 0 && lf >= 0, "Invalid boundary element mapping.");

            fes_scalar.GetElementDofs(el, el_dofs);
            const FiniteElement *fe = fes_scalar.GetFE(el);

            const IntegrationRule &nodes = fe->GetNodes();

            ElementTransformation &Tr = *fes_scalar.GetElementTransformation(el);


            for (int ld = 0; ld < fe->GetDof(); ld++)
            {
                const IntegrationPoint &ip = nodes.IntPoint(ld);
                Vector X(dim);
                Tr.Transform(ip, X);
                const double r = (dim > 1) ? X(1) : 0.0;

                if (std::abs(r) <= 1e-14)
                {
                    axis_idx.Append(el_dofs[ld]);
                }
             
            }
        }

        axis_idx.Sort();
        axis_idx.Unique();
    }

    void DGSEMOperator::ZeroAxisRadialMom(Vector &v) const
    {
        const int n = axis_idx.Size();
        
        if (n == 0) { return; }

        const int mom_r = 2;  // (rho, rho*uz. rho*ur, rhoE)
        auto *vr = v.GetData() + mom_r * num_dofs_scalar;

        for (int i = 0; i < n; i++)
        {
            vr[axis_idx[i]] = 0.0;
        }
    }

#endif

void DGSEMOperator::Mult(const Vector &u, Vector &dudt) const
{
#ifdef AXISYMMETRIC
    RecoverStateFromWeighted(u, U);

    const Vector &Ustate = U;
#else
    const Vector &Ustate = u;
#endif

#ifdef SUBCELL_FV_BLENDING
    ComputeBlendingCoefficient(Ustate);
#endif
      
#ifdef PARABOLIC
    ComputeGlobalEntropyVector(Ustate, global_entropy);

    if (dim == 1)
    {
        nonlinearForm->MultLifting(global_entropy, *grad_u[0]);
        ComputeGlobalPrimitiveGradVector(Ustate, *grad_u[0]);
        nonlinearForm->Mult(Ustate, *grad_u[0], dudt);
    }
    else if (dim == 2)
    {
      nonlinearForm->MultLifting(global_entropy, *grad_u[0], *grad_u[1]);
      ComputeGlobalPrimitiveGradVector(Ustate, *grad_u[0], *grad_u[1]);
      nonlinearForm->Mult(Ustate, *grad_u[0], *grad_u[1], dudt);    
    }
    else
    {
        nonlinearForm->MultLifting(global_entropy, *grad_u[0], *grad_u[1], *grad_u[2]);
        ComputeGlobalPrimitiveGradVector(Ustate, *grad_u[0], *grad_u[1], *grad_u[2]);
        nonlinearForm->Mult(Ustate, *grad_u[0], *grad_u[1], *grad_u[2], dudt);
    }

    #ifdef AXISYMMETRIC
        ZeroAxisRadialMom(dudt);
    #endif

#else

    max_char_speed = nonlinearForm->MultInviscid(Ustate, dudt);

    #ifdef AXISYMMETRIC
        ZeroAxisRadialMom(dudt);
    #endif

        // max_char_speed = integrator->GetMaxCharSpeed();
    for (int b = 0; b < bfnfi.size(); b++)
    {
        max_char_speed = std::max(bfnfi[b]->GetMaxCharSpeed(), max_char_speed);
    }
#endif

}

// void DGSEMOperator::AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker)
// {
// }

}
