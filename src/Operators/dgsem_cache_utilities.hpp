#pragma once
#include "mfem.hpp"
#include "dgsem_cache.hpp"

namespace Prandtl {

  template<typename CacheT>
  void GetOperatorCache(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    GetDiscretizationInfo(fes, cache);
    SetupRestrictions(fes, cache);
    SetupVolumeMarkers(fes, cache);
    SetupGeometricTerms(fes, cache);
 
    // TODO: Move these to where the caches are created and validated
    // MFEM_VERIFY(nfaces == cache.num_interior_faces, "restriction faces != cached interior faces");
    // MFEM_VERIFY(cache.face_normals.Size() == nfaces*nfp*dim, "normals size mismatch");
    // MFEM_VERIFY(cache.face_wt_minus.Size() == nfaces*nfp, "w_minus size mismatch");
    // MFEM_VERIFY(cache.face_wt_plus.Size()  == nfaces*nfp, "w_plus size mismatch");
  }

  template<typename GasModelT, typename DeviceCacheT>
  void SetupGasModel(GasModelT &gas_model, DeviceCacheT &device_cache)
  {
    device_cache.gas = gas_model;
  }

  template<typename CacheT>
  void GetDiscretizationInfo(mfem::FiniteElementSpace *fes, CacheT *cache)
  {

    MFEM_VERIFY(fes, "fes must be set");
    mfem::Mesh *mesh = fes->GetMesh();
    MFEM_VERIFY(mesh, "mesh must be set");
    const int p = fes->GetFE(0)->GetOrder();
    const int dim = mesh->SpaceDimension();
    const int Np = p + 1; // num 1d quadrature points
    const int Np_x = Np;
    const int Np_y = dim > 1 ? Np : 1;
    const int Np_z = dim > 2 ? Np : 1;
    const int ne = fes->GetNE();
    const int num_dofs_per_eqn_per_element = fes->GetFE(0)->GetDof();
    const int num_eqns = fes->GetVDim();

    cache->p = p;
    cache->Np = Np;
    cache->dim = dim;
    cache->Np_x = Np_x;
    cache->Np_y = Np_y;
    cache->Np_z = Np_z;
    cache->num_elements = ne;
    cache->ndof_scalar_el = num_dofs_per_eqn_per_element;
    cache->num_equations = num_eqns;
  }

  template<typename CacheT>
  void SetupRestrictions(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    MFEM_VERIFY(pfes, "Restriction setup requires ParFiniteElementSpace");
    cache->restr_v = fes->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
    cache->restr_f = pfes->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                              mfem::FaceType::Interior,
                                              mfem::L2FaceValues::DoubleValued);
  }

  // Set up and populate elJac, elMetric, D, Dhat, Dhat2
  // Face normals, and weights
  template<typename CacheT>
  void SetupGeometricTerms(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    const int nelem = cache->num_elements;
    const int p = cache->p;
    const int Np = cache->Np;
    const int dim = cache->dim;
    const int Np_x = Np;
    const int Np_y = dim > 1 ? Np : 1;
    const int Np_z = dim > 2 ? Np : 1;
    const int neq = cache->num_equations;
    mfem::Mesh *mesh = fes->GetMesh();

    // Build integration rules
    const int IntegrationOrder = 2 * Np_x - 3;
    cache->ir = &cache->GLIntRules.Get(mfem::Geometry::SEGMENT, IntegrationOrder);
    auto vol_topo = (dim == 1 ? mfem::Geometry::SEGMENT :
                     (dim == 2 ? mfem::Geometry::SQUARE : mfem::Geometry::CUBE));
    auto face_topo = (dim == 1 ? mfem::Geometry::POINT :
                      (dim == 2 ? mfem::Geometry::SEGMENT : mfem::Geometry::SQUARE));

    cache->ir_face = &cache->GLIntRules.Get(face_topo, IntegrationOrder);
    cache->ir_vol = &cache->GLIntRules.Get(vol_topo, IntegrationOrder);

    MFEM_ASSERT(cache->ir->GetNPoints() == Np_x, "");
    MFEM_ASSERT(cache->ir_vol->GetNPoints() == Np_x*Np_y*Np_z, "");

    // Populate element Jacobian determinant and metric terms
    cache->elJac.SetSize(Np_x*Np_y*Np_z*nelem);
    cache->elMetric.SetSize(dim*dim*Np_x*Np_y*Np_z*nelem);
    for (int i = 0; i < nelem; i++)
      {
        mfem::ElementTransformation *T = fes->GetElementTransformation(i);
        assert(T->ElementNo == i);
        AssembleElementVolumeGeometricTerms(*T, cache);
      }

    // Set up derivative operators
    mfem::DenseMatrix D_T, Dhat_T, Dhat2_T;
    D_T.SetSize(Np_x);
    Dhat_T.SetSize(Np_x);
    Dhat2_T.SetSize(Np_x);

    mfem::Vector wBary(Np_x);
    wBary = 1.0;

    for (int i = 1; i < Np_x; i++)
      {
        for (int j = 0; j < i; j++)
          {
            wBary(j) *= (cache->ir->IntPoint(j).x - cache->ir->IntPoint(i).x);
            wBary(i) *= (cache->ir->IntPoint(i).x - cache->ir->IntPoint(j).x);
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
                D_T(i, iL) = wBary(iL) / wBary(i) / (cache->ir->IntPoint(i).x - cache->ir->IntPoint(iL).x);
                D_T(i, i) -= D_T(i, iL);
              }
          }
      }

    Dhat_T = D_T;
    Dhat_T(0, 0) += 1.0 / cache->ir->IntPoint(0).weight;
    Dhat_T(Np - 1, Np - 1) -= 1.0 / cache->ir->IntPoint(Np - 1).weight;
    Dhat_T.Transpose();

    Dhat2_T = D_T;
    Dhat2_T *= 2.0;
    Dhat2_T(0, 0) += 1.0 / cache->ir->IntPoint(0).weight;
    Dhat2_T(Np - 1, Np - 1) -= 1.0 / cache->ir->IntPoint(Np - 1).weight;
    Dhat2_T.Transpose();
    D_T.Transpose();

    // Just copy D_T, Dhat_T, and Dhat2_T
    cache->D.SetSize(Np_x*Np_x);
    cache->Dhat.SetSize(Np_x*Np_x);
    cache->Dhat2.SetSize(Np_x*Np_x);
    std::memcpy(cache->D.HostWrite(),     D_T.Data(),     sizeof(real_t)*Np_x*Np_x);
    std::memcpy(cache->Dhat.HostWrite(),  Dhat_T.Data(),  sizeof(real_t)*Np_x*Np_x);
    std::memcpy(cache->Dhat2.HostWrite(), Dhat2_T.Data(), sizeof(real_t)*Np_x*Np_x);

    cache->elWaveSpeed.SetSize(nelem);
    cache->elWaveSpeed = 0.0;
    cache->elWaveSpeed.UseDevice();
    cache->elWaveSpeed.Read();

    cache->elJac.UseDevice();
    cache->elMetric.UseDevice();
    cache->D.UseDevice();
    cache->Dhat.UseDevice();
    cache->Dhat2.UseDevice();
    cache->elJac.Read();
    cache->elMetric.Read();
    cache->D.Read();
    cache->Dhat.Read();
    cache->Dhat2.Read();

    // Set up data for faces
    const int nfp = cache->ir_face->GetNPoints();
    cache->num_face_points = nfp;

    const int nfaces_restr = cache->restr_f->Height() / (nfp * neq * 2);
    cache->num_interior_faces = nfaces_restr;
    MFEM_VERIFY(nfaces_restr > 0, "nfaces_restr is 0");

    cache->face_normals.SetSize(nfaces_restr * nfp * dim);
    cache->face_wt_minus.SetSize(nfaces_restr * nfp);
    cache->face_wt_plus.SetSize(nfaces_restr * nfp);
    AssembleInteriorFaceGeometryTerms(fes, cache);
    
    cache->face_normals.UseDevice();
    cache->face_wt_minus.UseDevice();
    cache->face_wt_plus.UseDevice();
    cache->face_normals.Read();
    cache->face_wt_minus.Read();
    cache->face_wt_plus.Read();

    cache->ifWaveSpeed.SetSize(cache->num_interior_faces);
    cache->ifWaveSpeed = 0.0;
    cache->ifWaveSpeed.UseDevice();
    cache->ifWaveSpeed.Read();
  }


  template<typename CacheT>
  void SetupVolumeMarkers(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    mfem::Mesh *mesh = fes->GetMesh();
    
    cache->num_attr = mesh->attributes.Size() ? mesh->attributes.Max() : 0;
    cache->vol_attr_marker.SetSize(cache->num_attr);
    cache->vol_attr_marker = 1; // process everything

    cache->domain_attr_marker.SetSize(cache->num_attr);
    cache->domain_attr_marker = 1; // process everything
    
    // ---- 2) Per-element attribute id array -----------------------------------
    const int ne = mesh->GetNE();
    cache->elem_attr.SetSize(ne);
    for (int e = 0; e < ne; ++e)
      {
        const int attr = mesh->GetAttribute(e); // 1-based
        cache->elem_attr[e] = attr;
      }

    // Optional host-side sanity check (cheap, catches bad markers early):
    if (cache->num_attr > 0)
      {
        for (int e = 0; e < ne; ++e)
          {
            const int a = cache->elem_attr[e];
            MFEM_VERIFY(a >= 1 && a <= cache->num_attr,
                        "element attribute out of range: attr=" << a
                        << " num_attr=" << cache->num_attr);
          }
      }

    cache->elem_attr.UseDevice();
    cache->vol_attr_marker.UseDevice();
    cache->domain_attr_marker.UseDevice();
    cache->elem_attr.Read();
    cache->vol_attr_marker.Read();
    cache->domain_attr_marker.Read();
  }

  
  // Builds element-specific Jac/Metric and stuffs into cache.elJac, cache.elMetric
  template<typename CacheT>
  void AssembleElementVolumeGeometricTerms(mfem::ElementTransformation &Tr, CacheT *cache)
  {
    
    real_t *Jinv_h = cache->elJac.HostWrite();
    real_t *Met_h  = cache->elMetric.HostWrite();
    int dim = cache->dim;
    mfem::Vector metric1(dim);
    const int e = Tr.ElementNo;
    const int nq = cache->Np_x * cache->Np_y * cache->Np_z;
    
    for (int q = 0; q < nq; ++q)
      {
        const mfem::IntegrationPoint &ip = cache->ir_vol->IntPoint(q);
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

  //  void DGSEMNonlinearForm::AssembleFaceGeomCacheInterior()
  template<typename CacheT>
  void AssembleInteriorFaceGeometryTerms(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    auto *mesh = fes->GetMesh();
    auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    cache->fqs_int.reset(new mfem::FaceQuadratureSpace(*mesh, *cache->ir_face,
                                                       mfem::FaceType::Interior));
    MFEM_VERIFY(pfes, "need ParFiniteElementSpace");
    
    const int dim = mesh->Dimension();
    const int neq = pfes->GetVDim();
    const int nfp = cache->ir_face->GetNPoints();

    auto &int_faces = pmesh->GetFaceIndices(mfem::FaceType::Interior);
    const int ninterior_faces = int_faces.Size();

    cache->inv_fp_map.SetSize(ninterior_faces * nfp);    
    for (int face_slot = 0; face_slot < ninterior_faces; ++face_slot)
      {
        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            int fp_perm = cache->fqs_int->GetPermutedIndex(face_slot, fp_restr);
            cache->inv_fp_map[face_slot*nfp + fp_perm] = fp_restr;
          }
      }

    real_t *nor_d  = cache->face_normals.HostWrite();
    real_t *inv1_d = cache->face_wt_minus.HostWrite();
    real_t *inv2_d = cache->face_wt_plus.HostWrite();
    const real_t w0 = cache->ir->IntPoint(0).weight;
    
    auto store = [&](int fslot, int fp, const mfem::Vector &nor,
                     real_t inv_wJ1, real_t inv_wJ2)
    {
      const int nbase = (fslot * nfp + fp) * dim;
      for (int d = 0; d < dim; ++d) { nor_d[nbase + d] = nor(d); }
      inv1_d[fslot * nfp + fp] = inv_wJ1;
      inv2_d[fslot * nfp + fp] = inv_wJ2;
    };
    
    mfem::Vector nor(dim);
    const int num_elements_pmesh = pmesh->GetNE();
    // The order of faces in GetFaceIndices(FaceType::Interior) *must*
    // match the order of the faces in the interior face restriction
    // operator face slots.
    for (int fslot = 0; fslot < ninterior_faces; ++fslot)
      {
        const int face_id = int_faces[fslot];  
        bool face_is_flipped = false;
        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
            if (fp_geom != fp_restr){
              face_is_flipped = true;
            }
          }
        auto *tr = mesh->GetInteriorFaceTransformations(face_id);
        if (tr){ // Do interior face caching
          //          MFEM_VERIFY(tr, "expected interior face");
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
              const mfem::IntegrationPoint &ip = cache->ir_face->IntPoint(fp_geom);
              tr->SetAllIntPoints(&ip);
              
              const real_t J1 = tr->GetElement1Transformation().Weight();
              const real_t J2 = tr->GetElement2Transformation().Weight();
              
              if (dim == 1) { nor(0) = (tr->GetElement1IntPoint().x - 0.5)*2.0; }
              else          { mfem::CalcOrtho(tr->Jacobian(), nor); }
              
              //const real_t fac = face_is_flipped ? -1.0 : 1.0;
              const real_t fac = 1.0;
              store(fslot, fp_restr, nor, fac/(w0*J1), fac/(w0*J2));
            }
          continue;
        } // Internal face processing
        {
          auto *sh_tr = pmesh->GetSharedFaceTransformationsByLocalIndex(face_id, true);
          MFEM_VERIFY(sh_tr, "expected shared face");
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
              const mfem::IntegrationPoint &ip = cache->ir_face->IntPoint(fp_geom);
              sh_tr->SetAllIntPoints(&ip);
              
              const real_t J1 = sh_tr->GetElement1Transformation().Weight();
              const real_t J2 = sh_tr->GetElement2Transformation().Weight();
              
              if (dim == 1) { nor(0) = (sh_tr->GetElement1IntPoint().x - 0.5)*2.0; }
              else          { mfem::CalcOrtho(sh_tr->Jacobian(), nor); }
              
              //const real_t fac = face_is_flipped ? -1.0 : 1.0;
              const real_t fac1 = 1.0;
              const real_t fac2 = 0.0;
              store(fslot, fp_restr, nor, fac1/(w0*J1), fac2/(w0*J2));
            }
        } // Shared face processing
      }
  }

  template<typename CacheT, typename DeviceCacheT>
  void GetDeviceCache(CacheT &cache, DeviceCacheT &device_cache)
  {
    // Fixed data items
    device_cache.ndof_scalar_el = cache.ndof_scalar_el;
    device_cache.num_attr = cache.num_attr;
    device_cache.attr_marker_d = cache.vol_attr_marker.Read();
    device_cache.elem_attr_d = cache.elem_attr.Read();
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
    device_cache.nor_d = cache.face_normals.Read();
    device_cache.fw_minus_d = cache.face_wt_minus.Read();
    device_cache.fw_plus_d = cache.face_wt_plus.Read();

    // Updated every step by the compute device
    device_cache.elWaveSpeed_d = cache.elWaveSpeed.Write();
    device_cache.ifWaveSpeed_d = cache.ifWaveSpeed.Write();

    // POD gas model
    device_cache.gas = cache.gas;
    device_cache.iflux = cache.iflux;
 
  }

  template<typename CacheT>
  void OutputCacheContents(const CacheT &cache)
  {
    std::cout << "Cache Contents:" << std::endl
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

}
