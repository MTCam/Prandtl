#pragma once

#include "mfem.hpp"
#include "GasState.hpp"

namespace Prandtl
{
    using namespace mfem;

    real_t ComputeLogMean(real_t x, real_t y, real_t eps = 1e-4);

    inline real_t ComputeJump(real_t x, real_t y)
    {
        return (y - x);
    }

    inline real_t ComputeMean(real_t x, real_t y)
    {
        return 0.5 * (x + y);
    }

    void AddRow(DenseMatrix &A, const Vector &row, int r);

    void ComputeMean(const Vector &x, const Vector &y, Vector &mean);

    real_t ComputePressure(const Vector &state, real_t gammaM1);
    real_t ComputeEntropy(real_t rho, real_t p, real_t gamma);
    real_t ComputeInternalEnergy(real_t p, real_t rho, real_t gammaM1Inverse, real_t b = 0.0);
    real_t ComputeSoundSpeed(real_t p, real_t rho, real_t gamma, real_t b = 0.0);
    real_t ComputeEnthalpy(real_t p, real_t rho, real_t e);
    real_t ComputeTotalEnthalpy(const Vector &state, real_t gammaM1);

    void Conserv2Entropy(const DenseMatrix &vdof_mat, DenseMatrix &ent_mat, real_t gamma, real_t gammaM1, real_t gammaM1Inverse);
    void Conserv2Entropy(const Vector &state, Vector &ent_state, real_t gamma, real_t gammaM1, real_t gammaM1Inverse);
    void EntropyGrad2PrimGrad(const DenseMatrix &vdof_mat, DenseMatrix &grad, real_t gammaM1, real_t gammaM1Inverse);
    void Entropy2Conserv(const Vector &ent_state, Vector &state, real_t gamma, real_t gammaM1, real_t gammaM1Inverse);
    void Prim2Conserv(const Vector &state, Vector &conserv_state, real_t gammaM1Inverse);
    void Conserv2Prim(const Vector &state, Vector &prim_state, real_t gammaM1);

    inline void Normalize(Vector &vec)
    {
        vec /= vec.Norml2();
    }

  void Cross(const Vector &vec1, const Vector &vec2, Vector &cross);
  void Normal(const Vector &vec, Vector &nor);
  void RotateState(const StateLayout &layout, Vector &state, const Vector &nor);
  void RotateState(Vector &state, const Vector &nor);
  void RotateBack(Vector &state, const Vector &nor);
  void RotateBack(Vector &state, const Vector &nor, const StateLayout &layout);

    Vector ComputeRoeAverage(const Vector &state1, const Vector &state2, const real_t gamma);
  
    const Table& ElementIndextoBdrElementIndex(Mesh &mesh);
  
  template<typename GasModelT>
  inline void Conserv2Entropy(const GasModelT &gasModel, const Vector &state, Vector &ent_state)
  {
    PointStateView S{state.GetData()};
    PointStateViewRW E{ent_state.GetData()};
    gasModel.entropy_state(S, E);
  }
  
  template<typename GasModelT>
  inline void Conserv2Entropy(const GasModelT &gasModel, const DenseMatrix &vdof_mat, DenseMatrix &ent_mat)
  {
    ent_mat = 0.0;
    Vector state, ent_state(vdof_mat.Width());
    for (int d = 0; d < vdof_mat.Height(); d++)
      {
        vdof_mat.GetRow(d, state);
        Conserv2Entropy(gasModel, state, ent_state);
        ent_mat.SetRow(d, ent_state);
      }
  }
  
  template<typename GasModelT>
  inline void EntropyGrad2PrimGrad(const GasModelT &gasModel, const DenseMatrix &vdof_mat, DenseMatrix &grad)
  {
    Vector state, grad_state;
    
    int numeq = gasModel.num_equations();
    
    Vector gradPrim(numeq);
    
    Prandtl::PointStateViewRW dPrim{gradPrim.GetData()};
    
    for (int d = 0; d < vdof_mat.Height(); d++)
      {
        vdof_mat.GetRow(d, state);
        grad.GetRow(d, grad_state);
        Prandtl::PointStateView S{state.GetData()};
        Prandtl::PointStateView dS{grad_state.GetData()};
        gasModel.grad_entropy_to_grad_prim(S, dS, dPrim);
        grad.SetRow(d, gradPrim);
      }
  }
  
  template<typename GasModelT> 
  inline void Entropy2Conserv(const GasModelT &gasModel, const Vector &ent_state, Vector &state)
  {
    Prandtl::PointStateView Se{ent_state.GetData()};
    Prandtl::PointStateViewRW Sc{state.GetData()};
    gasModel.entropy_to_conserved(Se, Sc);
  }

}
