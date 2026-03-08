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
    AssembleDeviceCache();
    nonlinearForm->SetOperatorCache(&operator_cache);
    integrator->SetOperatorCache(&operator_cache);
  }
  
  void DGSEMOperator::CreateOperatorCache()
  {
    GetDiscretizationInfo(vfes.get(), &operator_cache);
    SetupRestrictions(vfes.get(), &operator_cache);
    SetupVolumeMarkers(vfes.get(), &operator_cache);
    SetupGeometricTerms(vfes.get(), &operator_cache);
  }

  void DGSEMOperator::AssembleDeviceCache()
  {
    // Copy the POD gasmodel outright 
    device_cache.gas = gasModel;
    GetDeviceCache(operator_cache, device_cache);
  }
  
  DGSEMOperator::~DGSEMOperator()
  {
    for (auto ptr : bfnfi)
      {
        delete ptr;
      }
  }
  
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

    nonlinearForm->Mult(Ustate, dudt);

    #ifdef AXISYMMETRIC
        ZeroAxisRadialMom(dudt);
    #endif

    max_char_speed = integrator->GetMaxCharSpeed();
    for (int b = 0; b < bfnfi.size(); b++)
    {
        max_char_speed = std::max(bfnfi[b]->GetMaxCharSpeed(), max_char_speed);
    }
#endif

}

void DGSEMOperator::AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker)
{
    nonlinearForm->AddBdrFaceIntegrator(bfi, bdr_marker);
}

}
