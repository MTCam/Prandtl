#include "DGSEMIntegrator.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{
  constexpr bool debug_integrator = false;
  
  DGSEMIntegrator::DGSEMIntegrator(std::shared_ptr<ParMesh> pmesh_,
                                   std::shared_ptr<ParFiniteElementSpace> fes0_,
                                   std::shared_ptr<ParGridFunction> alpha_,
                                   std::shared_ptr<LiftingScheme> liftingScheme_,
                                   NumericalFlux &rsolver_, int Np)
    : NonlinearFormIntegrator(), pmesh(pmesh_), fes0(fes0_), alpha(alpha_),
      liftingScheme(liftingScheme_), rsolver(rsolver_), fluxFunction(rsolver_.GetFluxFunction()),
      Np_x(Np), Np_y(fluxFunction.dim > 1 ? Np : 1), Np_z(fluxFunction.dim > 2 ? Np : 1),
      num_equations(fluxFunction.num_equations), dim(fluxFunction.dim), num_elements(pmesh->GetNE()),
      GLIntRules(0, Quadrature1D::GaussLobatto)
  {
    
    IntegrationOrder = 2 * Np_x - 3;
    ir = &GLIntRules.Get(Geometry::SEGMENT, IntegrationOrder);
    if (dim == 1)
    {
        ir_face = &GLIntRules.Get(Geometry::POINT, IntegrationOrder);
        ir_vol = &GLIntRules.Get(Geometry::SEGMENT, IntegrationOrder);
    }
    else if (dim == 2)
    {
        ir_face = &GLIntRules.Get(Geometry::SEGMENT, IntegrationOrder);
        ir_vol = &GLIntRules.Get(Geometry::SQUARE, IntegrationOrder);
    }
    else
    {
        ir_face = &GLIntRules.Get(Geometry::SQUARE, IntegrationOrder);
        ir_vol = &GLIntRules.Get(Geometry::CUBE, IntegrationOrder);
    }

  MFEM_ASSERT(ir->GetNPoints() == Np_x, "");
  MFEM_ASSERT(ir_vol->GetNPoints() == Np_x*Np_y*Np_z, "");

    max_char_speed = -1.0;
    
    D_T.SetSize(Np_x);
    Dhat_T.SetSize(Np_x);
    Dhat2_T.SetSize(Np_x);
    // Dhat2.SetSize(Np_x*Np_x);
 
    Vector wBary(Np_x);
    wBary = 1.0;

    for (int i = 1; i < Np_x; i++)
    {
        for (int j = 0; j < i; j++)
        {
            wBary(j) *= (ir->IntPoint(j).x - ir->IntPoint(i).x);
            wBary(i) *= (ir->IntPoint(i).x - ir->IntPoint(j).x);
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
                D_T(i, iL) = wBary(iL) / wBary(i) / (ir->IntPoint(i).x - ir->IntPoint(iL).x);
                D_T(i, i) -= D_T(i, iL);
            }
        }
    }
    
    Dhat_T = D_T;
    Dhat_T(0, 0) += 1.0 / ir->IntPoint(0).weight;
    Dhat_T(Np - 1, Np - 1) -= 1.0 / ir->IntPoint(Np - 1).weight;
    Dhat_T.Transpose();

    Dhat2_T = D_T;
    Dhat2_T *= 2.0;
    Dhat2_T(0, 0) += 1.0 / ir->IntPoint(0).weight;
    Dhat2_T(Np - 1, Np - 1) -= 1.0 / ir->IntPoint(Np - 1).weight;
    Dhat2_T.Transpose();
    // std::memcpy(Dhat2.GetData(), Dhat2_T.Data(), sizeof(real_t)*Np_x*Np_x);

    D_T.Transpose();

    dof = dof1 = dof2 = ir_vol->GetNPoints();
    
    shape1.SetSize(ir_vol->GetNPoints());
    shape2.SetSize(ir_vol->GetNPoints());


if (debug_integrator)
{
    std::cout << "[DGSEMIntegrator] Np = " << Np_x << ", IntegrationOrder = " << IntegrationOrder << ", dof = " << dof << "\n";
}
    state1.SetSize(num_equations);
    state2.SetSize(num_equations);

    f.SetSize(num_equations);
    g.SetSize(num_equations);
    h.SetSize(num_equations);

    flux_num.SetSize(num_equations);
    flux_mat1.SetSize(num_equations, dim);
    flux_mat2.SetSize(num_equations, dim);
    flux_mat.SetSize(dim, num_equations);

    flux_mat1 = flux_mat2 = 0.0;

    dqdx.SetSize(num_equations);
    dqdy.SetSize(num_equations);
    dqdz.SetSize(num_equations);

    el_dudxi.SetSize(num_equations);
    el_dudeta.SetSize(num_equations);
    el_dudzeta.SetSize(num_equations);

    adj1.SetSize(dim); adj2.SetSize(dim);
    metric1.SetSize(dim); metric2.SetSize(dim);
    nor.SetSize(dim);

    F_inviscid.SetSize(num_equations, Np_x, Np_x * Np_y * Np_z);
    G_inviscid.SetSize(num_equations, Np_y, Np_x * Np_y * Np_z);
    H_inviscid.SetSize(num_equations, Np_z, Np_x * Np_y * Np_z);
    
    F_viscous.SetSize(num_equations, Np_x * Np_y * Np_z);
    G_viscous.SetSize(num_equations, Np_x * Np_y * Np_z);
    H_viscous.SetSize(num_equations, Np_x * Np_y * Np_z);

    D_row.SetSize(Np_x);

    dU_subcell.SetSize(num_equations);
    dU_inviscid.SetSize(num_equations);
    dU_viscous.SetSize(num_equations);
    dU_volume.SetSize(num_equations);
    dU_face1.SetSize(num_equations);
    dU_face2.SetSize(num_equations);
    dU.SetSize(num_equations);


#ifdef PARABOLIC
    liftingScheme->SetLiftingParameters(ir, ir_face, ir_vol, D_T, num_equations, Np, dim);
#endif

#ifdef SUBCELL_FV_BLENDING
    SubcellMetricXi.SetSize(dim, Np_z * Np_y * (Np_x + 1), pmesh->GetNE());
    SubcellMetricEta.SetSize(dim, Np_z * (Np_y + 1) * Np_x, pmesh->GetNE());
    SubcellMetricZeta.SetSize(dim, (Np_z + 1) * Np_y * Np_x, pmesh->GetNE());
    ComputeSubcellMetrics();
#endif

    CreateOperatorCache();
}

  void DGSEMIntegrator::CreateOperatorCache() {
    cache.Np_x = Np_x;
    cache.Np_y = Np_y;
    cache.Np_z = Np_z;
    cache.dim = dim;
    cache.num_elements = fes0->GetNE();
    cache.num_equations = num_equations;
    AssembleGeometricTerms();
    cache.D.UseDevice();
    cache.Dhat.UseDevice();
    cache.Dhat2.UseDevice();
    cache.elJac.UseDevice();
    cache.elMetric.UseDevice();
    cache.D.Read();
    cache.Dhat.Read();
    cache.Dhat2.Read();
    cache.elJac.Read();
    cache.elMetric.Read();
  }

void DGSEMIntegrator::GetGeometricOperators(mfem::Vector &elJac_x, mfem::Vector &elMetric_x,
                                            mfem::Vector &D, mfem::Vector &Dhat,
                                            mfem::Vector &Dhat2)
{
  elJac_x.MakeRef(cache.elJac, 0, cache.elJac.Size());
  elMetric_x.MakeRef(cache.elMetric, 0, cache.elMetric.Size());

  D.SetSize(Np_x*Np_x);
  Dhat.SetSize(Np_x*Np_x);
  Dhat2.SetSize(Np_x*Np_x);

  std::memcpy(D.GetData(),     cache.D.GetData(),     sizeof(real_t)*Np_x*Np_x);
  std::memcpy(Dhat.GetData(),  cache.Dhat.GetData(),  sizeof(real_t)*Np_x*Np_x);
  std::memcpy(Dhat2.GetData(), cache.Dhat2.GetData(), sizeof(real_t)*Np_x*Np_x);
}

// Set up and populate elJac, elMetric, D, Dhat, Dhat2
void DGSEMIntegrator::AssembleGeometricTerms()
  {
    int nelem = fes0->GetNE();
    assert(nelem == num_elements);
    cache.elJac.SetSize(Np_x*Np_y*Np_z*num_elements);
    cache.elMetric.SetSize(dim*dim*Np_x*Np_y*Np_z*num_elements);

    cache.D.SetSize(Np_x*Np_x);
    cache.Dhat.SetSize(Np_x*Np_x);
    cache.Dhat2.SetSize(Np_x*Np_x);

    // Currently just copy the ole D_T, Dhat_T, and Dhat2_T
    std::memcpy(cache.D.GetData(),     D_T.Data(),     sizeof(real_t)*Np_x*Np_x);
    std::memcpy(cache.Dhat.GetData(),  Dhat_T.Data(),  sizeof(real_t)*Np_x*Np_x);
    std::memcpy(cache.Dhat2.GetData(), Dhat2_T.Data(), sizeof(real_t)*Np_x*Np_x);

    for (int i = 0; i < nelem; i++)
      {
        ElementTransformation *T = fes0->GetElementTransformation(i);
        assert(T->ElementNo == i);
        AssembleElementGeometricTerms(*T);
      }
  }

// Builds element-specific Jac/Metric and stuffs into cache.elJac, cache.elMetric
void DGSEMIntegrator::AssembleElementGeometricTerms(ElementTransformation &Tr)
{

  if (debug_integrator)
    {
      std::cout << "===== Entering DGSEMIntegrator::AssembleElementGeometricalTerms =====" << std::endl;
    }
  
  real_t *Jinv_h = cache.elJac.HostWrite();
  real_t *Met_h  = cache.elMetric.HostWrite();

  const int e = Tr.ElementNo;
  const int nq = Np_x * Np_y * Np_z;
  
  for (int q = 0; q < nq; ++q)
    {
      const IntegrationPoint &ip = ir_vol->IntPoint(q);
      Tr.SetIntPoint(&ip);
      const real_t J = Tr.Weight();
      Jinv_h[e*nq + q] = J;
      
      const mfem::DenseMatrix &adj = Tr.AdjugateJacobian();              
      for (int dir = 0; dir < dim; ++dir)
        {
          // adj.GetRow(dir, metricRow) pattern:
          // metricRow is a mfem::Vector of size dim you already have (e.g. member 'metric1')
          adj.GetRow(dir, metric1);  // metric1.Size() == dim
          
          for (int d = 0; d < dim; ++d)
            {
              const int idxM = (((e*nq + q)*dim + dir)*dim + d);
              Met_h[idxM] = metric1(d);
            }
        }
    }
  // cache.elJac.Read();
  // cache.elMetric.Read();
}

void DGSEMIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                         FaceElementTransformations &Tr, const Vector &el_u,
                                         Vector &el_dudt)
{
    if (debug_integrator) 
    {
        std::cout << "===== Entering DGSEMIntegrator::AssembleFaceVector =====" << std::endl;
    }
    el_dudt.SetSize((dof1 + dof2) * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);
    DenseMatrix el_dudt_mat2(el_dudt.GetData() + dof1 * num_equations, dof2, num_equations);

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

        if (dim == 1)
        {
            nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
        }
        else
        {
            CalcOrtho(Tr.Jacobian(), nor);
        }
        max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));

#ifdef AXISYMMETRIC
        Vector phys(dim);
        Tr.Transform(ip, phys);
        real_t r = phys[1]; 


        if (debug_integrator)
        {
            std::cout << "[AssembleFace]" << " Elem      =  " << Tr.ElementNo << "\n";
            std::cout << "[AssembleFace]" << " (x,r)     = (" << phys[0] << ", " << std::round(r) << ")" << "\n";
            std::cout << "[AssembleFace]" << " faceIP#   =  " << i <<", ip.x = " << ip.x << ", weight = " << ip.weight << ", J1 = " << J1 << ", J2 = " << J2 << "\n";
            std::cout << "[AssembleFace]" << " state1    = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
            std::cout << "[AssembleFace]" << " state2    = [" << state2[0] << ", " << state2[1] << ", " << state2[2] << ", " << state2[3] << "]\n";
            std::cout << "[AssembleFace]" << " flux_num  = [" << flux_num[0] << ", " << flux_num[1] << ", " << flux_num[2] << ", " << flux_num[3] << "]\n";
        }

        flux_num *= r;

         if (debug_integrator)
        {
            std::cout << "[AssembleFace]" << " rflux_num = [" << flux_num[0] << ", " << flux_num[1] << ", " << flux_num[2] << ", " << flux_num[3] << "]\n";
            std::cout << "________________________________________________________________________________________________________________" << "\n";
        }
#endif

   
        dU_face1 = dU_face2 = flux_num;
        dU_face1.Neg();

        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, dU_face2, el_dudt_mat2);
    }
}

void DGSEMIntegrator::AssembleElementVectorOG(const FiniteElement &el,
        ElementTransformation &Tr, const Vector &el_u, Vector &el_dudt)
{
    if (debug_integrator)
    {
        std::cout << "===== Entering DGSEMIntegrator::AssembleElementVector =====" << std::endl;
    }
    int e = Tr.ElementNo;
    int el_offset_jac = e*Np_x*Np_y*Np_z;
    int el_offset_metric = el_offset_jac*dim*dim;
    fes0->GetElementDofs(Tr.ElementNo, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);

    el_dudt.SetSize(dof * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt.GetData(), dof, num_equations);
#ifdef SUBCELL_FV_BLENDING    
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    
    for (int k = 0; k < Np_z; k++)
    {
        for (int j = 0; j < Np_y; j++)
        {
            for (int i = 0; i < Np_x; i++)
            {
                id1 = k * Np_y * Np_x + j * Np_x + i;
                const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
                Tr.SetIntPoint(&ip1);
                el_u_mat.GetRow(id1, state1);
                J = Tr.Weight();
                adj1 = Tr.AdjugateJacobian();
                adj1.GetRow(0, metric1);
                f = 0.0;
                F_inviscid(id1).SetCol(i, f);

                for (int m = i + 1; m < Np_x; m++)
                {
                    id2 = k * Np_y * Np_x + j * Np_x + m;
                    const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
                    Tr.SetIntPoint(&ip2);
                    el_u_mat.GetRow(id2, state2);
                    adj2 = Tr.AdjugateJacobian();
                    adj2.GetRow(0, metric2);
        
              
                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, f));
                
                IntegrationPoint ipm = ip1;
                ipm.x = 0.5*(ip1.x + ip2.x);
                Vector phys(dim);
                Tr.Transform(ipm, phys);
                real_t r_hat = phys[1]; 

                if (debug_integrator)
                {
                    std::cout << "[AssembleElement] [dim == 1]" << " (k, j, i, m) = (" << k << ", " << j << ", " << i << ", " << m << ")" << "\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " (id1, id2)   = (" << id1 << ", " << id2 << ")" << "\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " Elem         =  " << Tr.ElementNo << "\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " (x,r)        = (" << phys[0] << ", " << std::round(r_hat) << ")" << "\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " ip1.x.       =  " << ip1.x << ", w1 = " << ip1.weight << ", ip2.x = " << ip2.x << ", w2 = " << ip2.weight << ", J = " << J << "\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " state1.      = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " state2.      = [" << state2[0] << ", " << state2[1] << ", " << state2[2] << ", " << state2[3] << "]\n";
                    std::cout << "[AssembleElement] [dim == 1]" << " f            = [" << f[0] << ", " << f[1] << ", " << f[2] << ", " << f[3] << "]\n";
                }
              
#ifdef AXISYMMETRIC
OA                f *= r_hat;
#endif

                if (debug_integrator)
                {
                    std::cout << "[AssembleElement] [dim == 1]" << " rf           = [" << f[0] << ", " << f[1] << ", " << f[2] << ", " << f[3] << "]\n";
                }
                    F_inviscid(id1).SetCol(m, f);
                    F_inviscid(id2).SetCol(i, f);
                }

                if (dim > 1)
                {
                    adj1.GetRow(1, metric1);
                    g = 0.0;
                    G_inviscid(id1).SetCol(j, g);
                    for (int m = j + 1; m < Np_y; m++)
                    {
                        id2 = k * Np_y * Np_x + m * Np_x + i;
                        const IntegrationPoint &ip3 = ir_vol->IntPoint(id2);
                        Tr.SetIntPoint(&ip3);
                        el_u_mat.GetRow(id2, state2);  
                        adj2 = Tr.AdjugateJacobian();
                        adj2.GetRow(1, metric2);

                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, g));

                IntegrationPoint ipm = ip1;
                ipm.y = 0.5*(ip1.y + ip3.y);
                Vector phys(dim);
                Tr.Transform(ipm, phys);
                real_t r_hat = phys[1]; 

                if (debug_integrator)
                {
                    std::cout << "[AssembleElement] [dim == 2]" << " (k, j, i, m) = (" << k << ", " << j << ", " << i << ", " << m << ")" << "\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " (id1, id2)   = (" << id1 << ", " << id2 << ")" << "\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " Elem.        =  " << Tr.ElementNo << "\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " (x,r)        = (" << phys[0] << ", " << std::round(r_hat) << ")" << "\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " ip1.x        =  " << ip1.x << ", w1 = " << ip1.weight << ", ip3.x = " << ip3.x << ", w3 = " << ip3.weight << ", J = " << J << "\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " state1.      = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " state2.      = [" << state2[0] << ", " << state2[1] << ", " << state2[2] << ", " << state2[3] << "]\n";
                    std::cout << "[AssembleElement] [dim == 2]" << " g            = [" << g[0] << ", " << g[1] << ", " << g[2] << ", " << g[3] << "]\n";
                }

                
#ifdef AXISYMMETRIC
                g *= r_hat;
#endif

                if (debug_integrator)
                {
                    std::cout << "[AssembleElement] [dim == 2]" << " rg           = [" << g[0] << ", " << g[1] << ", " << g[2] << ", " << g[3] << "]\n";
                }
                        G_inviscid(id1).SetCol(m, g);
                        G_inviscid(id2).SetCol(j, g);
                    }
                    if (dim > 2)
                    {
                        adj1.GetRow(2, metric1);
                        h = 0.0;
                        H_inviscid(id1).SetCol(k, h);
                        for (int m = k + 1; m < Np_z; m++)
                        {
                            id2 = m * Np_y * Np_x + j * Np_x + i;
                            const IntegrationPoint &ip4 = ir_vol->IntPoint(id2);
                            el_u_mat.GetRow(id2, state2);
                            Tr.SetIntPoint(&ip4);
                            adj2 = Tr.AdjugateJacobian();
                            adj2.GetRow(2, metric2);
                            max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, h));
                            H_inviscid(id1).SetCol(m, h);
                            H_inviscid(id2).SetCol(k, h);
                        }
                    }
                }

                Dhat2_T.GetColumn(i, D_row); 
                F_inviscid(id1).Mult(D_row, dU_inviscid);

                if (debug_integrator)
                {
                    std::cout << "[AssembleElement] [dim == 1]" << " rdU_inviscid = [" << dU_inviscid[0] << ", " << dU_inviscid[1] << ", " << dU_inviscid[2] << ", " << dU_inviscid[3] << "]\n";
                }

                if (dim > 1)
                {
                    Dhat2_T.GetColumn(j, D_row);
                    G_inviscid(id1).AddMult(D_row, dU_inviscid);
                if (debug_integrator)
                {
                    std::cout << "[AssembleElement] [dim == 2]" << " rdU_inviscid = [" << dU_inviscid[0] << ", " << dU_inviscid[1] << ", " << dU_inviscid[2] << ", " << dU_inviscid[3] << "]\n";
                }
                    
                    if (dim > 2)
                    {
                        Dhat2_T.GetColumn(k, D_row);
                        H_inviscid(id1).AddMult(D_row, dU_inviscid);
                    }
                }

                 dU_inviscid.Neg();
                
#ifdef SUBCELL_FV_BLENDING
                dU_inviscid *= (1.0 - el_alpha(0));

                 if (debug_integrator)
                {
                    std::cout << "[AssembleElement]" << " alpha = " << el_alpha(0) << "\n";
                    std::cout << "[AssembleElement]" << " -rdU_inviscid *= 1-a    = [" << dU_inviscid[0] << ", " << dU_inviscid[1] << ", " << dU_inviscid[2] << ", " << dU_inviscid[3] << "]\n";
                }
#endif
                dU_inviscid /= J;

                if (debug_integrator)
                {
                    std::cout << "[AssembleElement]" << " -rdU_inviscid/J         = [" << dU_inviscid[0] << ", " << dU_inviscid[1] << ", " << dU_inviscid[2] << ", " << dU_inviscid[3] << "]\n";
                    std::cout << "________________________________________________________________________________________________________________" << "\n";
                }

                // TODO: Axisym source term is misplaced here. Need source term constructs
                //       to fix. It is a code smell that this bit of physics is leaking out to
                //       this level.  Currently, we are forced to make a utility in NS object
                //       to fetch the pressure to code around this breakage of the abstraction.
#ifdef AXISYMMETRIC
                
                {
                  const real_t p = fluxFunction.pressure(state1.GetData());
                  el_dudt_mat(id1, 2) += p;
                  
                  if (debug_integrator)
                    {
                      const DenseMatrix &Jm = Tr.Jacobian();
                      const real_t dr_deta_dbg = (Jm.NumRows() > 1 && Jm.NumCols() > 1) ? Jm(1,1) : 0.0;
                      std::cout << "[AssembleElement]" << " p = " << p << "\n"; //", r = " << r_hat << ", dr/deta = " << dr_deta_dbg << "\n";
                      std::cout << "[AssembleElement]" << " rdU_inviscid[2]-pJ      = [" << dU_inviscid[0] << ", " << dU_inviscid[1] << ", " << dU_inviscid[2] << ", " << dU_inviscid[3] << "]\n";
                    }
                  
                }            
#endif
                AddRow(el_dudt_mat, dU_inviscid, id1);
            }
        }
    }
}

void DGSEMIntegrator::AssembleElementVector(const FiniteElement &el,
                                            ElementTransformation &Tr,
                                            const Vector &el_u, Vector &el_dudt)
{
    if (debug_integrator)
    {
        std::cout << "===== Entering DGSEMIntegrator::AssembleElementVector =====" << std::endl;
    }

    int e = Tr.ElementNo;
    int el_offset_jac = e*Np_x*Np_y*Np_z;
    int el_offset_metric = el_offset_jac*dim*dim;
    const real_t *elJac_d = cache.elJac.Read() + el_offset_jac;
    const real_t *elMetric_d = cache.elMetric.Read() + el_offset_metric;
    const int N = Np_x;
    const real_t *Dhat2_d = cache.Dhat2.Read();

    fes0->GetElementDofs(Tr.ElementNo, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);

    el_dudt.SetSize(dof * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt.GetData(), dof, num_equations);

#ifdef SUBCELL_FV_BLENDING    
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    
    for (int k = 0; k < Np_z; k++)
      {
        for (int j = 0; j < Np_y; j++)
          {
            for (int i = 0; i < Np_x; i++)
              {
                id1 = k * Np_y * Np_x + j * Np_x + i;
                // const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
                // Tr.SetIntPoint(&ip1);
                el_u_mat.GetRow(id1, state1);
                // J = Tr.Weight();
                // double Jnew = elJac_d[id1];
                //double Jdiff = std::abs(J - Jnew);
                J = elJac_d[id1];
                //std::cout << "  Jerr: " << Jdiff << std::endl;
                // assert(Jdiff < 1e-12);
                // adj1 = Tr.AdjugateJacobian();
                // adj1.GetRow(0, metric1);
                real_t *m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim);
                met1.SetDataAndSize(m1ptr,dim);
                f = 0.0;
                F_inviscid(id1).SetCol(i, f);

                for (int m = i + 1; m < Np_x; m++)
                {
                    id2 = k * Np_y * Np_x + j * Np_x + m;
                    // const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
                    // Tr.SetIntPoint(&ip2);
                    el_u_mat.GetRow(id2, state2);
                    // adj2 = Tr.AdjugateJacobian();
                    // adj2.GetRow(0, metric2);
                    real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim);
                    met2.SetDataAndSize(m2ptr, dim);
              
                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, met1, met2, f));

                //IntegrationPoint ipm = ip1;
                //ipm.x = 0.5*(ip1.x + ip2.x);
                //Vector phys(dim);
                //Tr.Transform(ipm, phys);
                //real_t r_hat = phys[1]; 

                //#ifdef AXISYMMETRIC
                //                f *= r_hat;
                //#endif

                    F_inviscid(id1).SetCol(m, f);
                    F_inviscid(id2).SetCol(i, f);
                }

                if (dim > 1)
                {
                  // adj1.GetRow(1, metric1);
                  m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim+dim);
                  met1.SetDataAndSize(m1ptr,dim);
                    g = 0.0;
                    G_inviscid(id1).SetCol(j, g);
                    for (int m = j + 1; m < Np_y; m++)
                    {
                        id2 = k * Np_y * Np_x + m * Np_x + i;
                        // const IntegrationPoint &ip3 = ir_vol->IntPoint(id2);
                        // Tr.SetIntPoint(&ip3);
                        el_u_mat.GetRow(id2, state2);  
                        //adj2 = Tr.AdjugateJacobian();
                        // adj2.GetRow(1, metric2);
                        real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim+dim);
                        met2.SetDataAndSize(m2ptr, dim);

                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, met1, met2, g));

                // IntegrationPoint ipm = ip1;
                // ipm.y = 0.5*(ip1.y + ip3.y);
                // Vector phys(dim);
                // Tr.Transform(ipm, phys);
                // real_t r_hat = phys[1]; 
                // #ifdef AXISYMMETRIC
                // g *= r_hat;
                // #endif

                        G_inviscid(id1).SetCol(m, g);
                        G_inviscid(id2).SetCol(j, g);
                    }
                    if (dim > 2)
                    {
                      m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim+2*dim);
                      met1.SetDataAndSize(m1ptr,dim);
                      //  adj1.GetRow(2, metric1);
                        h = 0.0;
                        H_inviscid(id1).SetCol(k, h);
                        for (int m = k + 1; m < Np_z; m++)
                        {
                            id2 = m * Np_y * Np_x + j * Np_x + i;
                            //const IntegrationPoint &ip4 = ir_vol->IntPoint(id2);
                            el_u_mat.GetRow(id2, state2);
                            // Tr.SetIntPoint(&ip4);
                            // adj2 = Tr.AdjugateJacobian();
                            //  adj2.GetRow(2, metric2);
                            real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim+2*dim);
                            met2.SetDataAndSize(m2ptr, dim);
                            max_char_speed = std::max(max_char_speed,
                                                      rsolver.ComputeVolumeFlux(state1, state2, met1, met2, h));
                            H_inviscid(id1).SetCol(m, h);
                            H_inviscid(id2).SetCol(k, h);
                        }
                    }
                }
                real_t *dhat_ptr = const_cast<real_t*>(Dhat2_d + i*N);
                D_row.SetDataAndSize(dhat_ptr, N);
                // Dhat2_T.GetColumn(i, D_row); 
                F_inviscid(id1).Mult(D_row, dU_inviscid);

                if (dim > 1)
                {
                  //Dhat2_T.GetColumn(j, D_row);
                  dhat_ptr = const_cast<real_t*>(Dhat2_d + j*N);
                  D_row.SetDataAndSize(dhat_ptr, N);
                    G_inviscid(id1).AddMult(D_row, dU_inviscid);
                    if (dim > 2)
                    {
                      dhat_ptr = const_cast<real_t*>(Dhat2_d + k*N);
                      D_row.SetDataAndSize(dhat_ptr, N);
                      //Dhat2_T.GetColumn(k, D_row);
                        H_inviscid(id1).AddMult(D_row, dU_inviscid);
                    }
                }

                 dU_inviscid.Neg();
                
#ifdef SUBCELL_FV_BLENDING
                dU_inviscid *= (1.0 - el_alpha(0));
#endif
                dU_inviscid /= J;

#ifdef AXISYMMETRIC
                
                {
                  Prandtl::PointStateView S{state1.GetData(), stateLayout.get()};
                  const real_t p = gasModel.pressure(S);
                    //dU_inviscid(2) += p;
                    el_dudt_mat(id1, 2) += p; 
                }            
#endif
                AddRow(el_dudt_mat, dU_inviscid, id1);
            }
        }
      }
}

real_t DGSEMIntegrator::AssembleElementVectorHost(const FiniteElement &el,
                                                  ElementTransformation &Tr,
                                                  const Vector &el_u, Vector &el_dudt)
{
    if (debug_integrator)
    {
        std::cout << "===== Entering DGSEMIntegrator::AssembleElementVector =====" << std::endl;
    }

    int e = Tr.ElementNo;
    int el_offset_jac = e*Np_x*Np_y*Np_z;
    int el_offset_metric = el_offset_jac*dim*dim;
    const real_t *elJac_d = cache.elJac.Read() + el_offset_jac;
    const real_t *elMetric_d = cache.elMetric.Read() + el_offset_metric;
    const int N = Np_x;
    const real_t *Dhat2_d = cache.Dhat2.Read();

    fes0->GetElementDofs(Tr.ElementNo, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);

    el_dudt.SetSize(dof * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt.GetData(), dof, num_equations);

#ifdef SUBCELL_FV_BLENDING    
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    
    for (int k = 0; k < Np_z; k++)
      {
        for (int j = 0; j < Np_y; j++)
          {
            for (int i = 0; i < Np_x; i++)
              {
                id1 = k * Np_y * Np_x + j * Np_x + i;
                // const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
                // Tr.SetIntPoint(&ip1);
                el_u_mat.GetRow(id1, state1);
                // J = Tr.Weight();
                // double Jnew = elJac_d[id1];
                //double Jdiff = std::abs(J - Jnew);
                J = elJac_d[id1];
                //std::cout << "  Jerr: " << Jdiff << std::endl;
                // assert(Jdiff < 1e-12);
                // adj1 = Tr.AdjugateJacobian();
                // adj1.GetRow(0, metric1);
                real_t *m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim);
                met1.SetDataAndSize(m1ptr,dim);
                f = 0.0;
                F_inviscid(id1).SetCol(i, f);

                for (int m = i + 1; m < Np_x; m++)
                {
                    id2 = k * Np_y * Np_x + j * Np_x + m;
                    // const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
                    // Tr.SetIntPoint(&ip2);
                    el_u_mat.GetRow(id2, state2);
                    // adj2 = Tr.AdjugateJacobian();
                    // adj2.GetRow(0, metric2);
                    real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim);
                    met2.SetDataAndSize(m2ptr, dim);
              
                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, met1, met2, f));

                //IntegrationPoint ipm = ip1;
                //ipm.x = 0.5*(ip1.x + ip2.x);
                //Vector phys(dim);
                //Tr.Transform(ipm, phys);
                //real_t r_hat = phys[1]; 

                //#ifdef AXISYMMETRIC
                //                f *= r_hat;
                //#endif

                    F_inviscid(id1).SetCol(m, f);
                    F_inviscid(id2).SetCol(i, f);
                }

                if (dim > 1)
                {
                  // adj1.GetRow(1, metric1);
                  m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim+dim);
                  met1.SetDataAndSize(m1ptr,dim);
                    g = 0.0;
                    G_inviscid(id1).SetCol(j, g);
                    for (int m = j + 1; m < Np_y; m++)
                    {
                        id2 = k * Np_y * Np_x + m * Np_x + i;
                        // const IntegrationPoint &ip3 = ir_vol->IntPoint(id2);
                        // Tr.SetIntPoint(&ip3);
                        el_u_mat.GetRow(id2, state2);  
                        //adj2 = Tr.AdjugateJacobian();
                        // adj2.GetRow(1, metric2);
                        real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim+dim);
                        met2.SetDataAndSize(m2ptr, dim);

                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, met1, met2, g));

                // IntegrationPoint ipm = ip1;
                // ipm.y = 0.5*(ip1.y + ip3.y);
                // Vector phys(dim);
                // Tr.Transform(ipm, phys);
                // real_t r_hat = phys[1]; 
                // #ifdef AXISYMMETRIC
                // g *= r_hat;
                // #endif

                        G_inviscid(id1).SetCol(m, g);
                        G_inviscid(id2).SetCol(j, g);
                    }
                    if (dim > 2)
                    {
                      m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim+2*dim);
                      met1.SetDataAndSize(m1ptr,dim);
                      //  adj1.GetRow(2, metric1);
                        h = 0.0;
                        H_inviscid(id1).SetCol(k, h);
                        for (int m = k + 1; m < Np_z; m++)
                        {
                            id2 = m * Np_y * Np_x + j * Np_x + i;
                            //const IntegrationPoint &ip4 = ir_vol->IntPoint(id2);
                            el_u_mat.GetRow(id2, state2);
                            // Tr.SetIntPoint(&ip4);
                            // adj2 = Tr.AdjugateJacobian();
                            //  adj2.GetRow(2, metric2);
                            real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim+2*dim);
                            met2.SetDataAndSize(m2ptr, dim);
                            max_char_speed = std::max(max_char_speed,
                                                      rsolver.ComputeVolumeFlux(state1, state2, met1, met2, h));
                            H_inviscid(id1).SetCol(m, h);
                            H_inviscid(id2).SetCol(k, h);
                        }
                    }
                }
                real_t *dhat_ptr = const_cast<real_t*>(Dhat2_d + i*N);
                D_row.SetDataAndSize(dhat_ptr, N);
                // Dhat2_T.GetColumn(i, D_row); 
                F_inviscid(id1).Mult(D_row, dU_inviscid);

                if (dim > 1)
                {
                  //Dhat2_T.GetColumn(j, D_row);
                  dhat_ptr = const_cast<real_t*>(Dhat2_d + j*N);
                  D_row.SetDataAndSize(dhat_ptr, N);
                    G_inviscid(id1).AddMult(D_row, dU_inviscid);
                    if (dim > 2)
                    {
                      dhat_ptr = const_cast<real_t*>(Dhat2_d + k*N);
                      D_row.SetDataAndSize(dhat_ptr, N);
                      //Dhat2_T.GetColumn(k, D_row);
                        H_inviscid(id1).AddMult(D_row, dU_inviscid);
                    }
                }

                 dU_inviscid.Neg();
                
#ifdef SUBCELL_FV_BLENDING
                dU_inviscid *= (1.0 - el_alpha(0));
#endif
                dU_inviscid /= J;

#ifdef AXISYMMETRIC
                
                {
                  Prandtl::PointStateView S{state1.GetData(), stateLayout.get()};
                  const real_t p = gasModel.pressure(S);
                    //dU_inviscid(2) += p;
                    el_dudt_mat(id1, 2) += p; 
                }            
#endif
                AddRow(el_dudt_mat, dU_inviscid, id1);
            }
        }
      }
    return max_char_speed;
}

// This function is a TEST that makes sure that our caller (DGSEMNonlinearForm)
// using element restriction operator instead of ElementTransformation Tr works
// and gets us the correct data in the correct shape.
real_t DGSEMIntegrator::AssembleElementVolumeHost(const int e,ElementTransformation &Tr,
                                                  const real_t *el_u, real_t *el_dudt)
{
    if (debug_integrator)
    {
        std::cout << "===== Entering DGSEMIntegrator::AssembleElementVolumeHost =====" << std::endl;
    }

    // int e = Tr.ElementNo;
    const int Np_x = cache.Np_x;
    const int Np_y = cache.Np_y;
    const int Np_z = cache.Np_z;
    const int dim = cache.dim;
    const int dof = Np_x * Np_y * Np_z;
    const int num_equations = cache.num_equations;
    int el_offset_jac = e*Np_x*Np_y*Np_z;
    int el_offset_metric = el_offset_jac*dim*dim;
    const real_t *elJac_d = cache.elJac.Read() + el_offset_jac;
    const real_t *elMetric_d = cache.elMetric.Read() + el_offset_metric;
    const int N = Np_x;
    const real_t *Dhat2_d = cache.Dhat2.Read();
    for (int q = 0;q < dof * num_equations;q++){
      el_dudt[q] = 0.0;
    }
    // el_dudt.SetSize(dof * num_equations);
    // el_dudt = 0.0;

    const DenseMatrix el_u_mat(const_cast<real_t*>(el_u), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt, dof, num_equations);

#ifdef SUBCELL_FV_BLENDING
    fes0->GetElementDofs(e, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    
    for (int k = 0; k < Np_z; k++)
      {
        for (int j = 0; j < Np_y; j++)
          {
            for (int i = 0; i < Np_x; i++)
              {
                id1 = k * Np_y * Np_x + j * Np_x + i;
                // const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
                // Tr.SetIntPoint(&ip1);
                el_u_mat.GetRow(id1, state1);
                // J = Tr.Weight();
                // double Jnew = elJac_d[id1];
                //double Jdiff = std::abs(J - Jnew);
                J = elJac_d[id1];
                //std::cout << "  Jerr: " << Jdiff << std::endl;
                // assert(Jdiff < 1e-12);
                // adj1 = Tr.AdjugateJacobian();
                // adj1.GetRow(0, metric1);
                real_t *m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim);
                met1.SetDataAndSize(m1ptr,dim);
                f = 0.0;
                F_inviscid(id1).SetCol(i, f);

                for (int m = i + 1; m < Np_x; m++)
                {
                    id2 = k * Np_y * Np_x + j * Np_x + m;
                    // const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
                    // Tr.SetIntPoint(&ip2);
                    el_u_mat.GetRow(id2, state2);
                    // adj2 = Tr.AdjugateJacobian();
                    // adj2.GetRow(0, metric2);
                    real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim);
                    met2.SetDataAndSize(m2ptr, dim);
              
                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, met1, met2, f));

                //IntegrationPoint ipm = ip1;
                //ipm.x = 0.5*(ip1.x + ip2.x);
                //Vector phys(dim);
                //Tr.Transform(ipm, phys);
                //real_t r_hat = phys[1]; 

                //#ifdef AXISYMMETRIC
                //                f *= r_hat;
                //#endif

                    F_inviscid(id1).SetCol(m, f);
                    F_inviscid(id2).SetCol(i, f);
                }

                if (dim > 1)
                {
                  // adj1.GetRow(1, metric1);
                  m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim+dim);
                  met1.SetDataAndSize(m1ptr,dim);
                    g = 0.0;
                    G_inviscid(id1).SetCol(j, g);
                    for (int m = j + 1; m < Np_y; m++)
                    {
                        id2 = k * Np_y * Np_x + m * Np_x + i;
                        // const IntegrationPoint &ip3 = ir_vol->IntPoint(id2);
                        // Tr.SetIntPoint(&ip3);
                        el_u_mat.GetRow(id2, state2);  
                        //adj2 = Tr.AdjugateJacobian();
                        // adj2.GetRow(1, metric2);
                        real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim+dim);
                        met2.SetDataAndSize(m2ptr, dim);

                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, met1, met2, g));

                // IntegrationPoint ipm = ip1;
                // ipm.y = 0.5*(ip1.y + ip3.y);
                // Vector phys(dim);
                // Tr.Transform(ipm, phys);
                // real_t r_hat = phys[1]; 
                // #ifdef AXISYMMETRIC
                // g *= r_hat;
                // #endif

                        G_inviscid(id1).SetCol(m, g);
                        G_inviscid(id2).SetCol(j, g);
                    }
                    if (dim > 2)
                    {
                      m1ptr = const_cast<real_t *>(elMetric_d+id1*dim*dim+2*dim);
                      met1.SetDataAndSize(m1ptr,dim);
                      //  adj1.GetRow(2, metric1);
                        h = 0.0;
                        H_inviscid(id1).SetCol(k, h);
                        for (int m = k + 1; m < Np_z; m++)
                        {
                            id2 = m * Np_y * Np_x + j * Np_x + i;
                            //const IntegrationPoint &ip4 = ir_vol->IntPoint(id2);
                            el_u_mat.GetRow(id2, state2);
                            // Tr.SetIntPoint(&ip4);
                            // adj2 = Tr.AdjugateJacobian();
                            //  adj2.GetRow(2, metric2);
                            real_t *m2ptr = const_cast<real_t *>(elMetric_d+id2*dim*dim+2*dim);
                            met2.SetDataAndSize(m2ptr, dim);
                            max_char_speed = std::max(max_char_speed,
                                                      rsolver.ComputeVolumeFlux(state1, state2, met1, met2, h));
                            H_inviscid(id1).SetCol(m, h);
                            H_inviscid(id2).SetCol(k, h);
                        }
                    }
                }
                real_t *dhat_ptr = const_cast<real_t*>(Dhat2_d + i*N);
                D_row.SetDataAndSize(dhat_ptr, N);
                // Dhat2_T.GetColumn(i, D_row); 
                F_inviscid(id1).Mult(D_row, dU_inviscid);

                if (dim > 1)
                {
                  //Dhat2_T.GetColumn(j, D_row);
                  dhat_ptr = const_cast<real_t*>(Dhat2_d + j*N);
                  D_row.SetDataAndSize(dhat_ptr, N);
                    G_inviscid(id1).AddMult(D_row, dU_inviscid);
                    if (dim > 2)
                    {
                      dhat_ptr = const_cast<real_t*>(Dhat2_d + k*N);
                      D_row.SetDataAndSize(dhat_ptr, N);
                      //Dhat2_T.GetColumn(k, D_row);
                        H_inviscid(id1).AddMult(D_row, dU_inviscid);
                    }
                }

                 dU_inviscid.Neg();
                
#ifdef SUBCELL_FV_BLENDING
                 dU_inviscid *= (1.0 - el_alpha(0));
#endif
                dU_inviscid /= J;

#ifdef AXISYMMETRIC
                
                {
                  Prandtl::PointStateView S{state1.GetData(), stateLayout.get()};
                  const real_t p = gasModel.pressure(S);
                    //dU_inviscid(2) += p;
                    el_dudt_mat(id1, 2) += p; 
                }            
#endif
                AddRow(el_dudt_mat, dU_inviscid, id1);
            }
        }
      }
    return max_char_speed;
}

void DGSEMIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                         FaceElementTransformations &Tr, const Vector &el_u,
                                         const Vector &el_dudx, const Vector &el_dudy,
                                         const Vector &el_dudz, Vector &el_dudt)
{
    el_dudt.SetSize((dof1 + dof2) * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);

    const DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    const DenseMatrix el_dudx_mat2(el_dudx.GetData() + dof1 * num_equations, dof2, num_equations);

    const DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
    const DenseMatrix el_dudy_mat2(el_dudy.GetData() + dof1 * num_equations, dof2, num_equations);

    const DenseMatrix el_dudz_mat1(el_dudz.GetData(), dof1, num_equations);
    const DenseMatrix el_dudz_mat2(el_dudz.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);
    DenseMatrix el_dudt_mat2(el_dudt.GetData() + dof1 * num_equations, dof2, num_equations);

    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir_face->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        J1 = Tr.GetElement1Transformation().Weight();
        J2 = Tr.GetElement2Transformation().Weight();
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

        el_u_mat1.MultTranspose(shape1, state1);
        el_dudx_mat1.MultTranspose(shape1, dqdx);
        el_dudy_mat1.MultTranspose(shape1, dqdy);
        el_dudz_mat1.MultTranspose(shape1, dqdz);
        fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, dqdz, flux_mat1);

        el_u_mat2.MultTranspose(shape2, state2);
        el_dudx_mat2.MultTranspose(shape2, dqdx);
        el_dudy_mat2.MultTranspose(shape2, dqdy);
        el_dudz_mat2.MultTranspose(shape2, dqdz);   
        fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, dqdz, flux_mat2);

        CalcOrtho(Tr.Jacobian(), nor);

        max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));
        dU_face1 = dU_face2 = flux_num;
        dU_face1.Neg();

        flux_mat1 += flux_mat2;
        flux_mat1 *= 0.5;
        flux_mat1.Mult(nor, flux_num);
        dU_face1 += flux_num;
        dU_face2 -= flux_num;  

        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, dU_face2, el_dudt_mat2);
    }
}

void DGSEMIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                         FaceElementTransformations &Tr, const Vector &el_u,
                                         const Vector &el_dudx, const Vector &el_dudy, Vector &el_dudt)
{
    el_dudt.SetSize((dof1 + dof2) * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);

    const DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    const DenseMatrix el_dudx_mat2(el_dudx.GetData() + dof1 * num_equations, dof2, num_equations);

    const DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
    const DenseMatrix el_dudy_mat2(el_dudy.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);
    DenseMatrix el_dudt_mat2(el_dudt.GetData() + dof1 * num_equations, dof2, num_equations);

    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir_face->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        J1 = Tr.GetElement1Transformation().Weight();
        J2 = Tr.GetElement2Transformation().Weight();
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

        el_u_mat1.MultTranspose(shape1, state1);
        el_dudx_mat1.MultTranspose(shape1, dqdx);
        el_dudy_mat1.MultTranspose(shape1, dqdy);
        fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, flux_mat1);

        el_u_mat2.MultTranspose(shape2, state2);
        el_dudx_mat2.MultTranspose(shape2, dqdx);
        el_dudy_mat2.MultTranspose(shape2, dqdy);
        fluxFunction.ComputeViscousFlux(state2, dqdx, dqdy, flux_mat2);

        CalcOrtho(Tr.Jacobian(), nor);

        max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));
        dU_face1 = dU_face2 = flux_num;
        dU_face1.Neg();

        flux_mat1 += flux_mat2;
        flux_mat1 *= 0.5;
        flux_mat1.Mult(nor, flux_num);
        dU_face1 += flux_num;
        dU_face2 -= flux_num;  

        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, dU_face2, el_dudt_mat2);
    }
}

void DGSEMIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                         FaceElementTransformations &Tr, const Vector &el_u,
                                         const Vector &el_dudx, Vector &el_dudt)
{
    el_dudt.SetSize((dof1 + dof2) * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
    const DenseMatrix el_u_mat2(el_u.GetData() + dof1 * num_equations, dof2, num_equations);

    const DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
    const DenseMatrix el_dudx_mat2(el_dudx.GetData() + dof1 * num_equations, dof2, num_equations);
    
    DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);
    DenseMatrix el_dudt_mat2(el_dudt.GetData() + dof1 * num_equations, dof2, num_equations);

    for (int i = 0; i < ir_face->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir_face->IntPoint(i);
        Tr.SetAllIntPoints(&ip);
        J1 = Tr.GetElement1Transformation().Weight();
        J2 = Tr.GetElement2Transformation().Weight();
        el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
        el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

        el_u_mat1.MultTranspose(shape1, state1);
        el_dudx_mat1.MultTranspose(shape1, dqdx);
        fluxFunction.ComputeViscousFlux(state1, dqdx, flux_mat1);

        el_u_mat2.MultTranspose(shape2, state2);
        el_dudx_mat2.MultTranspose(shape2, dqdx);
        fluxFunction.ComputeViscousFlux(state2, dqdx, flux_mat2);  

        nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;

        max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));
        dU_face1 = dU_face2 = flux_num;
        dU_face1.Neg();

        flux_mat1 += flux_mat2;
        flux_mat1 *= 0.5;
        flux_mat1.Mult(nor, flux_num);
        dU_face1 += flux_num;
        dU_face2 -= flux_num;  

        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
        AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J2), shape2, dU_face2, el_dudt_mat2);
    }
}

void DGSEMIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, const Vector &el_dudz, Vector &el_dudt)
{
    fes0->GetElementDofs(Tr.ElementNo, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);

    el_dudt.SetSize(dof * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    const DenseMatrix el_dudx_mat(el_dudx.GetData(), dof, num_equations);
    const DenseMatrix el_dudy_mat(el_dudy.GetData(), dof, num_equations);
    const DenseMatrix el_dudz_mat(el_dudz.GetData(), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt.GetData(), dof, num_equations);
#ifdef SUBCELL_FV_BLENDING
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    for (int i = 0; i < ir_vol->GetNPoints(); i++)
    {
        const IntegrationPoint &ip1 = ir_vol->IntPoint(i);
        el_u_mat.GetRow(id1, state1);
        Tr.SetIntPoint(&ip1);
        adj1 = Tr.AdjugateJacobian();

        el_dudx_mat.GetRow(i, dqdx);

        el_dudy_mat.GetRow(i, dqdy);

        el_dudz_mat.GetRow(i, dqdz);      

        fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, dqdz, flux_mat1);

        mfem::MultABt(adj1, flux_mat1, flux_mat);
        flux_mat.GetRow(0, f);
        F_viscous.SetCol(i, f);

        flux_mat.GetRow(1, g);
        G_viscous.SetCol(i, g);

        flux_mat.GetRow(2, h);
        H_viscous.SetCol(i, h);        
    }
    
    for (int k = 0; k < Np_z; k++)
    {
        for (int j = 0; j < Np_y; j++)
        {
            for (int i = 0; i < Np_x; i++)
            {
                id1 = k * Np_y * Np_x + j * Np_x + i;
                const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
                el_u_mat.GetRow(id1, state1);
                Tr.SetIntPoint(&ip1);
                J = Tr.Weight();
                adj1 = Tr.AdjugateJacobian();

                adj1.GetRow(0, metric1);
                f = 0.0;
                F_inviscid(id1).SetCol(i, f);

                for (int m = i + 1; m < Np_x; m++)
                {
                    id2 = k * Np_y * Np_x + j * Np_x + m;
                    const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
                    el_u_mat.GetRow(id2, state2);
                    Tr.SetIntPoint(&ip2);
                    adj2 = Tr.AdjugateJacobian();
                    adj2.GetRow(0, metric2);
                    max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, f));
                    F_inviscid(id1).SetCol(m, f);
                    F_inviscid(id2).SetCol(i, f);
                }

                adj1.GetRow(1, metric1);
                g = 0.0;
                G_inviscid(id1).SetCol(j, g);
                for (int m = j + 1; m < Np_y; m++)
                {
                    id2 = k * Np_y * Np_x + m * Np_x + i;
                    const IntegrationPoint &ip3 = ir_vol->IntPoint(id2);
                    el_u_mat.GetRow(id2, state2);
                    Tr.SetIntPoint(&ip3);
                    adj2 = Tr.AdjugateJacobian();
                    adj2.GetRow(1, metric2);
                    max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, g));
                    G_inviscid(id1).SetCol(m, g);
                    G_inviscid(id2).SetCol(j, g);
                }

                adj1.GetRow(2, metric1);
                h = 0.0;
                H_inviscid(id1).SetCol(k, h);
                for (int m = k + 1; m < Np_z; m++)
                {
                    id2 = m * Np_y * Np_x + j * Np_x + i;
                    const IntegrationPoint &ip4 = ir_vol->IntPoint(id2);
                    el_u_mat.GetRow(id2, state2);
                    Tr.SetIntPoint(&ip4);
                    adj2 = Tr.AdjugateJacobian();
                    adj2.GetRow(2, metric2);
                    max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, h));
                    H_inviscid(id1).SetCol(m, h);
                    H_inviscid(id2).SetCol(k, g);
                }

                dU_viscous = 0.0;

                Dhat2_T.GetColumn(i, D_row); 
                F_inviscid(id1).Mult(D_row, dU_inviscid);

                Dhat_T.GetColumn(i, D_row);
                for (int l = 0; l < Np_x; l++)
                {
                    F_viscous.GetColumn(k * Np_x * Np_y + j * Np_x + l, dU);
                    dU *= D_row(l);
                    dU_viscous += dU;
                }

                Dhat2_T.GetColumn(j, D_row);
                G_inviscid(id1).AddMult(D_row, dU_inviscid);
                
                Dhat_T.GetColumn(j, D_row);
                for (int l = 0; l < Np_y; l++)
                {
                    G_viscous.GetColumn(k * Np_x * Np_y + l * Np_x + i, dU);
                    dU *= D_row(l);
                    dU_viscous += dU;
                }

                Dhat2_T.GetColumn(k, D_row);
                H_inviscid(id1).AddMult(D_row, dU_inviscid);

                Dhat_T.GetColumn(k, D_row);
                for (int l = 0; l < Np_z; l++)
                {
                    H_viscous.GetColumn(l * Np_x * Np_y + j * Np_x + i, dU);
                    dU *= D_row(l);
                    dU_viscous += dU;
                }
         
                dU_inviscid.Neg();
#ifdef SUBCELL_FV_BLENDING
                dU_inviscid *= (1.0 - el_alpha(0));
#endif
                add(dU_inviscid, dU_viscous, dU_volume);
                dU_volume /= J;
                AddRow(el_dudt_mat, dU_volume, id1);
            }
        }
    }    
}

void DGSEMIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, Vector &el_dudt)
{
    fes0->GetElementDofs(Tr.ElementNo, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);

    el_dudt.SetSize(dof * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    const DenseMatrix el_dudx_mat(el_dudx.GetData(), dof, num_equations);
    const DenseMatrix el_dudy_mat(el_dudy.GetData(), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt.GetData(), dof, num_equations);
#ifdef SUBCELL_FV_BLENDING
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    for (int i = 0; i < ir_vol->GetNPoints(); i++)
    {
        const IntegrationPoint &ip1 = ir_vol->IntPoint(i);
        el_u_mat.GetRow(id1, state1);
        Tr.SetIntPoint(&ip1);
        adj1 = Tr.AdjugateJacobian();

        el_dudx_mat.GetRow(i, dqdx);

        el_dudy_mat.GetRow(i, dqdy);

        fluxFunction.ComputeViscousFlux(state1, dqdx, dqdy, flux_mat1);

        mfem::MultABt(adj1, flux_mat1, flux_mat);
        flux_mat.GetRow(0, f);
        F_viscous.SetCol(i, f);

        flux_mat.GetRow(1, g);
        G_viscous.SetCol(i, g);
    }
    
    for (int j = 0; j < Np_y; j++)
    {
        for (int i = 0; i < Np_x; i++)
        {
            id1 = j * Np_x + i;
            const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
            el_u_mat.GetRow(id1, state1);
            Tr.SetIntPoint(&ip1);
            J = Tr.Weight();
            adj1 = Tr.AdjugateJacobian();
            adj1.GetRow(0, metric1);
            f = 0.0;
            F_inviscid(id1).SetCol(i, f);

            for (int m = i + 1; m < Np_x; m++)
            {
                id2 = j * Np_x + m;
                const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
                el_u_mat.GetRow(id2, state2);
                Tr.SetIntPoint(&ip2);
                adj2 = Tr.AdjugateJacobian();
                adj2.GetRow(0, metric2);
                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, f));
                F_inviscid(id1).SetCol(m, f);
                F_inviscid(id2).SetCol(i, f);
            }

            adj1.GetRow(1, metric1);
            g = 0.0;
            G_inviscid(id1).SetCol(j, g);
            for (int m = j + 1; m < Np_y; m++)
            {
                id2 = m * Np_x + i;
                const IntegrationPoint &ip3 = ir_vol->IntPoint(id2);
                el_u_mat.GetRow(id2, state2);
                Tr.SetIntPoint(&ip3);
                adj2 = Tr.AdjugateJacobian();
                adj2.GetRow(1, metric2);
                max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, g));
                G_inviscid(id1).SetCol(m, g);
                G_inviscid(id2).SetCol(j, g);
            }

            dU_viscous = 0.0;

            Dhat2_T.GetColumn(i, D_row); 
            F_inviscid(id1).Mult(D_row, dU_inviscid);

            Dhat_T.GetColumn(i, D_row);
            for (int l = 0; l < Np_x; l++)
            {
                F_viscous.GetColumn(j * Np_x + l, dU);
                dU *= D_row(l);
                dU_viscous += dU;
            }

            Dhat2_T.GetColumn(j, D_row);
            G_inviscid(id1).AddMult(D_row, dU_inviscid);
            
            Dhat_T.GetColumn(j, D_row);
            for (int l = 0; l < Np_y; l++)
            {
                G_viscous.GetColumn(l * Np_x + i, dU);
                dU *= D_row(l);
                dU_viscous += dU;
            }
        
            dU_inviscid.Neg();
#ifdef SUBCELL_FV_BLENDING
            dU_inviscid *= (1.0 - el_alpha(0));
#endif
            add(dU_inviscid, dU_viscous, dU_volume);
            dU_volume /= J;
            AddRow(el_dudt_mat, dU_volume, id1);
        }
    }   
}

void DGSEMIntegrator::AssembleElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, const Vector &el_dudx, Vector &el_dudt)
{
    fes0->GetElementDofs(Tr.ElementNo, alpha_indx);
    alpha->GetSubVector(alpha_indx, el_alpha);

    el_dudt.SetSize(dof * num_equations);
    el_dudt = 0.0;

    const DenseMatrix el_u_mat(el_u.GetData(), dof, num_equations);
    const DenseMatrix el_dudx_mat(el_dudx.GetData(), dof, num_equations);
    DenseMatrix el_dudt_mat(el_dudt.GetData(), dof, num_equations);
#ifdef SUBCELL_FV_BLENDING
    ComputeFVFluxes(el_u_mat, el_alpha(0), Tr, el_dudt_mat);
#endif
    for (int i = 0; i < ir_vol->GetNPoints(); i++)
    {
        const IntegrationPoint &ip1 = ir_vol->IntPoint(i);
        el_u_mat.GetRow(id1, state1);
        Tr.SetIntPoint(&ip1);
        adj1 = Tr.AdjugateJacobian();

        el_dudx_mat.GetRow(i, dqdx);

        fluxFunction.ComputeViscousFlux(state1, dqdx, flux_mat1);

        mfem::MultABt(adj1, flux_mat1, flux_mat);
        flux_mat.GetRow(0, f);
        F_viscous.SetCol(i, f);
    }
    
    for (int i = 0; i < Np_x; i++)
    {
        const IntegrationPoint &ip1 = ir_vol->IntPoint(id1);
        el_u_mat.GetRow(id1, state1);
        Tr.SetIntPoint(&ip1);
        J = Tr.Weight();
        adj1 = Tr.AdjugateJacobian();
        adj1.GetRow(0, metric1);
        f = 0.0;
        F_inviscid(id1).SetCol(i, f);

        for (int m = i + 1; m < Np_x; m++)
        {
            const IntegrationPoint &ip2 = ir_vol->IntPoint(id2);
            el_u_mat.GetRow(id2, state2);
            Tr.SetIntPoint(&ip2);
            adj2 = Tr.AdjugateJacobian();
            adj2.GetRow(0, metric2);
            max_char_speed = std::max(max_char_speed, rsolver.ComputeVolumeFlux(state1, state2, metric1, metric2, f));
            F_inviscid(i).SetCol(m, f);
            F_inviscid(m).SetCol(i, f);
        }

        dU_viscous = 0.0;

        Dhat2_T.GetColumn(i, D_row); 
        F_inviscid(id1).Mult(D_row, dU_inviscid);

        Dhat_T.GetColumn(i, D_row);
        for (int l = 0; l < Np_x; l++)
        {
            F_viscous.GetColumn(l, dU);
            dU *= D_row(l);
            dU_viscous += dU;
        }
    
        dU_inviscid.Neg();
#ifdef SUBCELL_FV_BLENDING
        dU_inviscid *= (1.0 - el_alpha(0));
#endif
        add(dU_inviscid, dU_viscous, dU_volume);
        dU_volume /= J;
        AddRow(el_dudt_mat, dU_volume, id1);
    }   
}

void DGSEMIntegrator::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz)
{
    liftingScheme->AssembleLiftingFaceVector(el1, el2, Tr, el_u, el_dudx, el_dudy, el_dudz);
}

void DGSEMIntegrator::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy)
{
    liftingScheme->AssembleLiftingFaceVector(el1, el2, Tr, el_u, el_dudx, el_dudy);
}

void DGSEMIntegrator::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx)
{
    liftingScheme->AssembleLiftingFaceVector(el1, el2, Tr, el_u, el_dudx);
}

void DGSEMIntegrator::AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz)
{
    liftingScheme->AssembleLiftingElementVector(el, Tr, el_u, el_dudx, el_dudy, el_dudz);
}

void DGSEMIntegrator::AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy)
{
    liftingScheme->AssembleLiftingElementVector(el, Tr, el_u, el_dudx, el_dudy);
}

void DGSEMIntegrator::AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx)
{
    liftingScheme->AssembleLiftingElementVector(el, Tr, el_u, el_dudx);
}

void DGSEMIntegrator::ComputeFVFluxes(const DenseMatrix &el_u_mat, real_t alpha_value, ElementTransformation &Tr, DenseMatrix &el_dudt_mat)
{
    if (debug_integrator)
    {
        std::cout << "===== Entering DGSEMIntegrator::ComputeFVFluxes =====" << std::endl;
    }
    
    auto mid_r = [&](int idL, int idR, int dir) -> real_t
    {
        IntegrationPoint ipL = ir_vol->IntPoint(idL);
        IntegrationPoint ipR = ir_vol->IntPoint(idR);

        IntegrationPoint ipm = ipL;
        if (dir == 0) {
            ipm.x = 0.5 * (ipL.x + ipR.x);
        }
        else if (dir == 1) {
            ipm.y = 0.5 * (ipL.y + ipR.y);
        }
        else {
            ipm.z = 0.5 * (ipL.z + ipR.z);
        }

        Vector phys(dim);
        Tr.Transform(ipm, phys);

        return phys[1]; 
    };

    for (int k = 0; k < Np_z; k++)
    {
        for (int j = 0; j < Np_y; j++)
        {
            dU_subcell = 0.0;
            id1 = k * Np_y * Np_x + j * Np_x;
            el_u_mat.GetRow(id1, state1);
            for (int i = 0; i < Np_x - 1; i++)
            {
                id2 = id1 + 1;
                el_u_mat.GetRow(id2, state2);
                SubcellMetricXi(Tr.ElementNo).GetColumn(id2, nor);
                max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));

                real_t r_hat  = mid_r(id1, id2, 0);

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " (k, j, i)               = (" << k << ", " << j << ", " << i << ")" << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " (id1, id2)              = (" << id1 << ", " << id2 << ")" << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " Elem                    =  " << Tr.ElementNo << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " r                       =  " << std::round(r_hat) << ", faceIP# = " << i << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " nor                     = (" << nor[0] << ", " << nor[1] << ")\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " state1                  = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " state2                  = [" << state2[0] << ", " << state2[1] << ", " << state2[2] << ", " << state2[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " flux_num                = [" << flux_num[0] << ", " << flux_num[1] << ", " << flux_num[2] << ", " << flux_num[3] << "]\n";
                }          

               
#ifdef AXISYMMETRIC
                flux_num *= r_hat;
#endif

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " rflux_num               = [" << flux_num[0] << ", " << flux_num[1] << ", " << flux_num[2] << ", " << flux_num[3] << "]\n";
                }
                Tr.SetIntPoint(&ir_vol->IntPoint(id1));
                dU_subcell -= flux_num;

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " dU_subcell -= rfluxnum  = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                }

                dU_subcell /= Tr.Weight() * (ir->IntPoint(i).weight);

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " J = " << Tr.Weight() << ", w = " << ir->IntPoint(i).weight << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " dU_subcell =/w.J        = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                }

                el_dudt_mat.SetRow(id1, dU_subcell);

                dU_subcell = flux_num;
                state1 = state2;
                id1 = id2;

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " dU_subcell = rflux_num  = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " state1 = state2         = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " id1 = id2               = " << id1 << "\n";
                }
            }
            Tr.SetIntPoint(&ir_vol->IntPoint(id1));
            dU_subcell /= Tr.Weight() * (ir->IntPoint(Np_x - 1).weight);

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " J = " << Tr.Weight() << ", w = " << ir->IntPoint(Np_x - 1).weight << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 1]  [Xi]" << " dU_subcell =/w.J        = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                }

            el_dudt_mat.SetRow(id1, dU_subcell);
        }
    }

    if (dim > 1)
    {
        for (int k = 0; k < Np_z; k++)
        {
            for (int i = 0; i < Np_x; i++)
            {
                dU_subcell = 0.0;
                id1 = k * Np_y * Np_x + i;
                el_u_mat.GetRow(id1, state1);
                for (int j = 0; j < Np_y - 1; j++)
                {
                    id2 = k * Np_y * Np_x + (j + 1) * Np_x + i;
                    el_u_mat.GetRow(id2, state2);
                    SubcellMetricEta(Tr.ElementNo).GetColumn(id2, nor);
                    max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));

                    real_t r_hat  = mid_r(id1, id2, 1);
                    
                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " (k, i, j)               = (" << k << ", " << i << ", " << j << ")" << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " (id1, id2)              = (" << id1 << ", " << id2 << ")" << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " Elem                    =  " << Tr.ElementNo << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " r                       =  " << std::round(r_hat) << ", faceIP# = " << j << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " nor                     = (" << nor[0] << ", " << nor[1] << ")\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " state1                  = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " state2                  = [" << state2[0] << ", " << state2[1] << ", " << state2[2] << ", " << state2[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " fluxnum                 = [" << flux_num[0] << ", " << flux_num[1] << ", " << flux_num[2] << ", " << flux_num[3] << "]\n";
                }  
              
#ifdef AXISYMMETRIC
                flux_num *= r_hat;
#endif

                  if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " rfluxnum                = [" << flux_num[0] << ", " << flux_num[1] << ", " << flux_num[2] << ", " << flux_num[3] << "]\n";
                }

                Tr.SetIntPoint(&ir_vol->IntPoint(id1));
                dU_subcell -= flux_num;

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " dU_subcell -= rflux_num = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                }

                dU_subcell /= Tr.Weight() * (ir->IntPoint(j).weight);

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " J                       = " << Tr.Weight() << ", w = " << ir->IntPoint(j).weight << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " dU_subcell =/w.J        = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                }

                    AddRow(el_dudt_mat, dU_subcell, id1);

                    dU_subcell = flux_num;
                    state1 = state2;
                    id1 = id2;   
                    
                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " dU_subcell = rflux_num  = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " state1 = state2         = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " id1 = id2               = " << id1 << "\n";
                }
                    
                }
                Tr.SetIntPoint(&ir_vol->IntPoint(id1));
                dU_subcell /= Tr.Weight() * (ir->IntPoint(Np_y - 1).weight);

                if (debug_integrator)
                {
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " J = " << Tr.Weight() << ", w = " << ir->IntPoint(Np_y - 1).weight << "\n";
                    std::cout << "[ComputeFVFluxes] [dim == 2] [Eta]" << " dU_subcell =/w.J        = [" << dU_subcell[0] << ", " << dU_subcell[1] << ", " << dU_subcell[2] << ", " << dU_subcell[3] << "]\n";
                    std::cout << "________________________________________________________________________________________________________________" << "\n";
                }

                AddRow(el_dudt_mat, dU_subcell, id1);
            }
        }

        if (dim > 2)
        {
            for (int j = 0; j < Np_y; j++)
            {
                for (int i = 0; i < Np_x; i++)
                {
                    dU_subcell = 0.0;
                    id1 = j * Np_x + i;
                    el_u_mat.GetRow(id1, state1);
                    for (int k = 0; k < Np_z - 1; k++)
                    {
                        id2 = (k + 1) * Np_y * Np_x + j * Np_x + i;
                        el_u_mat.GetRow(id2, state2);
                        SubcellMetricZeta(Tr.ElementNo).GetColumn(id2, nor);
                        max_char_speed = std::max(max_char_speed, rsolver.ComputeFaceFlux(state1, state2, nor, flux_num));
                        Tr.SetIntPoint(&ir_vol->IntPoint(id1));

                        dU_subcell -= flux_num;
                        dU_subcell /= Tr.Weight() * (ir->IntPoint(k).weight);
                        AddRow(el_dudt_mat, dU_subcell, id1);

                        dU_subcell = flux_num;
                        state1 = state2;
                        id1 = id2;            
                    }
                    Tr.SetIntPoint(&ir_vol->IntPoint(id1));
                    dU_subcell /= Tr.Weight() * (ir->IntPoint(Np_z - 1).weight);
                    AddRow(el_dudt_mat, dU_subcell, id1);
                }
            }
        }
    }
    el_dudt_mat *= alpha_value;
}


void DGSEMIntegrator::ComputeSubcellMetrics()
{
    for (int el = 0; el < pmesh->GetNE(); el++)
    {
        ElementTransformation *Tr = pmesh->GetElementTransformation(el);
        DenseMatrix &nor_mat_xi = SubcellMetricXi(el);
        DenseMatrix &nor_mat_eta = SubcellMetricEta(el);
        DenseMatrix &nor_mat_zeta = SubcellMetricZeta(el);
        Vector tmp(dim);
        for (int k = 0; k < Np_z; k++)
        {
            for (int j = 0; j < Np_y; j++)
            {
                const IntegrationPoint &ip = ir_vol->IntPoint(k * Np_y * Np_x + j * Np_x); // left xi-face
                Tr->SetIntPoint(&ip);
                Tr->AdjugateJacobian().GetRow(0, metric1);
                for (int i = 0; i < Np_x + 1; i++)
                {
                    nor = metric1;
                    for (int l = 0; l < i; l++)
                    {
                        D_T.GetColumn(l, D_row);
                        tmp = 0.0;
                        real_t weight = ir->IntPoint(l).weight;
                        for (int m = 0; m < Np_x; m++)
                        {
                            Tr->SetIntPoint(&ir_vol->IntPoint(k * Np_y * Np_x + j * Np_x + m));
                            Tr->AdjugateJacobian().GetRow(0, metric2);
                            metric2 *= D_row(m);
                            tmp += metric2;
                        }
                        tmp *= weight;
                        nor += tmp;
                    }
                    nor_mat_xi.SetCol(k * Np_y * Np_x + j * Np_x + i, nor);
                }
            }
        }

        if (dim > 1)
        {
            for (int k = 0; k < Np_z; k++)
            {
                for (int i = 0; i < Np_x; i++)
                {
                    const IntegrationPoint &ip = ir_vol->IntPoint(k * Np_y * Np_x + i); // bottom eta-face
                    Tr->SetIntPoint(&ip);
                    Tr->AdjugateJacobian().GetRow(1, metric1);
                    for (int j = 0; j < Np_y + 1; j++)
                    {
                        nor = metric1;
                        for (int l = 0; l < j; l++)
                        {
                            D_T.GetColumn(l, D_row);
                            tmp = 0.0;
                            real_t weight = ir->IntPoint(l).weight;
                            for (int m = 0; m < Np_y; m++)
                            {
                                Tr->SetIntPoint(&ir_vol->IntPoint(k * Np_y * Np_x + m * Np_x + i));
                                Tr->AdjugateJacobian().GetRow(1, metric2);
                                metric2 *= D_row(m);
                                tmp += metric2;
                            }
                            tmp *= weight;
                            nor += tmp;
                        }
                        nor_mat_eta.SetCol(k * Np_y * Np_x + j * Np_x + i, nor);
                    }
                }
            }

            if (dim > 2)
            {
                for (int j = 0; j < Np_y; j++)
                {
                    for (int i = 0; i < Np_x; i++)
                    {
                        const IntegrationPoint &ip = ir_vol->IntPoint(j * Np_x + i); // bottom zeta-face
                        Tr->SetIntPoint(&ip);
                        Tr->AdjugateJacobian().GetRow(2, metric1);
                        for (int k = 0; k < Np_z + 1; k++)
                        {
                            nor = metric1;
                            for (int l = 0; l < k; l++)
                            {
                                D_T.GetColumn(l, D_row);
                                tmp = 0.0;
                                real_t weight = ir->IntPoint(l).weight;
                                for (int m = 0; m < Np_z; m++)
                                {
                                    Tr->SetIntPoint(&ir_vol->IntPoint(m * Np_y * Np_x + j * Np_x + i));
                                    Tr->AdjugateJacobian().GetRow(2, metric2);
                                    metric2 *= D_row(m);
                                    tmp += metric2;
                                }
                                tmp *= weight;
                                nor += tmp;
                            }
                            nor_mat_zeta.SetCol(k * Np_y * Np_x + j * Np_x + i, nor);
                        }
                    }
                }
            }
        }
    }   
}

}
