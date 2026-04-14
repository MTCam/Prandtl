#pragma once

#include "DGSEMIntegrator.hpp"
#include "DGSEMNonlinearForm.hpp"
#include "BdrFaceIntegrator.hpp"
#include "ModalBasis.hpp"
#include "Indicator.hpp"
#include "BasicOperations.hpp"
#include "GasModel.hpp"
#include "dgsem_cache_utilities.hpp"

namespace Prandtl
{

using namespace mfem;
  
class DGSEMOperator : public TimeDependentOperator
{
private:
    std::shared_ptr<ParFiniteElementSpace> vfes;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::shared_ptr<ParMesh> pmesh;
    std::shared_ptr<ParGridFunction> eta;
    std::shared_ptr<ParGridFunction> alpha;
    std::vector<std::shared_ptr<ParGridFunction> > grad_u;
    std::shared_ptr<ParGridFunction> r_gf;
    std::unique_ptr<DGSEMIntegrator> integrator;
    std::unique_ptr<Indicator> indicator;
    const ActiveGasModel gasModel;
    std::unique_ptr<DGSEMNonlinearForm> nonlinearForm; 

    mutable Array<int> vdof_indices;
    mutable Vector el_vdofs, grad_vdofs;

    const int num_equations, dim, order, num_elements;
    const int num_dofs_scalar;
    const int Ndofs;

    mutable Vector global_entropy;
    
#ifdef AXISYMMETRIC
    mutable long long calls_accum = 0, highOrder_shape_accum = 0, lowOrder_ray2_accum = 0, lowOrder_ray1_accum = 0, lowOrder_copy_accum = 0;
    mutable Vector U;
    Array<int> axis_marker;
    Array<int> axis_idx;
    bool low_order_axis = false;
    real_t rho_floor_abs = 1e-12;
    real_t p_floor_abs = 1e-12;
    bool highOrder_near_axis_toggle = false;
    real_t highOrder_near_axis_fraction = 0.5; // 0.0 -> 100% highOrder, 1.0 -> 100% lowOrder
#endif
    
    const real_t sharpness_fac = 9.21024;
    const real_t modalThreshold;
    const real_t alpha_min;
    const real_t alpha_max;

    mutable real_t max_char_speed;
    
    std::vector<BdrFaceIntegrator*> bfnfi;
    std::vector<Array<int>> bdr_marker;
    mutable Array<int> ind_indx;
    mutable Vector ind_dof;
    mutable real_t alpha_dof;
    DGSEMOperatorCache operator_cache;
    DGSEMDeviceCache device_cache;

    void CreateOperatorCache();
    void ComputeGlobalEntropyVector(const Vector &u, Vector &global_entropy) const;
    void ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx) const;
    void ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy) const;
    void ComputeGlobalPrimitiveGradVector(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const;
    void ComputeBlendingCoefficient(const Vector &u) const;
    void AssembleDeviceCache();

#ifdef AXISYMMETRIC
    void BuildAxisIndexFromMarker();
    void ZeroAxisRadialMom(Vector &v) const;
    void ResetAxisReconStats() const
    {
        calls_accum = highOrder_shape_accum = lowOrder_ray2_accum = lowOrder_ray1_accum = lowOrder_copy_accum = 0;
    }

    struct AxisReconStats {
        long long calls, highOrder_shape, lowOrder_ray2, lowOrder_ray1, lowOrder_copy;
    };
#endif

public:
    DGSEMOperator(std::shared_ptr<ParFiniteElementSpace> vfes,
                  std::shared_ptr<ParFiniteElementSpace> fes0,
                  std::shared_ptr<ParMesh> pmesh,
                  std::shared_ptr<ParGridFunction> eta,
                  std::shared_ptr<ParGridFunction> alpha,
                  std::vector<std::shared_ptr<ParGridFunction> > &grad_u_,
                  std::unique_ptr<DGSEMIntegrator> integrator,
                  std::unique_ptr<Indicator> indicator,
                  const ActiveGasModel &gasModel_,
                  std::shared_ptr<ParGridFunction> r_gf = nullptr,
                  const real_t alpha_max = 0.5, const real_t alpha_min = 0.001);
    
    ~DGSEMOperator();
    
    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker);
    
    void Mult(const Vector &u, Vector &dudt) const override;
    inline real_t GetMaxCharSpeed()
    {
        return max_char_speed;
    }

    inline real_t& GetTimeRef()
    {
        return t;
    }

#ifdef AXISYMMETRIC
    void RecoverStateFromWeighted(const Vector &rU, Vector &U) const;
    long long GetTotalFallbacks1st(bool global = true) const;
    long long GetTotalFallbacks0th(bool global = true) const;
    AxisReconStats GetAxisReconStats(bool global = true) const;
    void SetAxisBoundaryMarker(const Array<int>& marker) 
    { 
        axis_marker = marker;
        BuildAxisIndexFromMarker();
    }
    void SetLowOrderAxis(bool enable) 
    {
        low_order_axis = enable;
    }
    bool GetLowOrderAxis() const
    {
        return low_order_axis;
    }
    void SetAxisFloorsFromFreestream(real_t rho_inf, real_t p_inf, real_t rho_fac = 1e-8, real_t p_fac = 1e-8)
    {
        rho_floor_abs = std::max(rho_floor_abs, rho_fac * rho_inf);
        p_floor_abs = std::max(p_floor_abs, p_fac * p_inf);
    }
#endif
};

}
