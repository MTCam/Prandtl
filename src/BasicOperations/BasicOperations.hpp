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
  
  template<typename GasModelT, typename TabStruct>
  inline void Conserv2Entropy(const GasModelT &gasModel, const TabStruct &thermoTables, const Vector &state, Vector &ent_state)
  {
    PointStateView S{state.GetData()};
    PointStateViewRW E{ent_state.GetData()};
    gasModel.entropy_state(S, E, thermoTables);
  }
  
  template<typename GasModelT, typename TabStruct>
  inline void Conserv2Entropy(const GasModelT &gasModel, const TabStruct &thermoTables, const DenseMatrix &vdof_mat, DenseMatrix &ent_mat)
  {
    ent_mat = 0.0;
    Vector state, ent_state(vdof_mat.Width());
    for (int d = 0; d < vdof_mat.Height(); d++)
      {
        vdof_mat.GetRow(d, state);
        Conserv2Entropy(gasModel, thermoTables, state, ent_state);
        ent_mat.SetRow(d, ent_state);
      }
  }
  
  template<typename GasModelT, typename TabStruct>
  inline void EntropyGrad2PrimGrad(const GasModelT &gasModel, const TabStruct &thermoTables,const DenseMatrix &vdof_mat, DenseMatrix &grad)
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
        gasModel.grad_entropy_to_grad_prim(S, dS, dPrim, thermoTables);
        grad.SetRow(d, gradPrim);
      }
  }
  
  template<typename GasModelT, typename TabStruct>
  inline void Entropy2Conserv(const GasModelT &gasModel, const TabStruct &thermoTables, const Vector &ent_state, Vector &state)
  {
    Prandtl::PointStateView Se{ent_state.GetData()};
    Prandtl::PointStateViewRW Sc{state.GetData()};
    gasModel.entropy_to_conserved(Se, Sc, thermoTables);
  }

  MFEM_HOST_DEVICE inline
  int hunt(const real_t *arr, int n, real_t x, int ind_lo)
  {
      int ind_hi, ind_mid;
      int incr = 1;
      bool ascend = (arr[n-1] >= arr[0]);

      if (ind_lo < 0 || ind_lo >= n)
      {
          ind_lo = -1;
          ind_hi =  n;
      }
      else
      {
          // Right or Left Hunt
          if ( (x >= arr[ind_lo]) == ascend)
          {
              // Hunt right
              if (ind_lo == n-1) return ind_lo;
              ind_hi = ind_lo + incr;

              while (ind_hi < n && ((x >= arr[ind_hi]) == ascend))
              {
                  ind_lo = ind_hi;
                  incr *= 2;
                  ind_hi = ind_lo + incr;
                  if (ind_hi > n-1)
                  {
                      ind_hi = n;
                      break;
                  }
              }
          }
          // Hunt left
          else
          {
              if (ind_lo == 0)
              {
                  ind_lo = -1;
                  return ind_lo;
              }
              ind_hi = ind_lo;
              ind_lo = ind_lo-1;
              while (ind_lo >= 0 && ((x < arr[ind_lo]) == ascend))
              {
                  ind_hi = ind_lo;
                  incr *= 2;
                  if (incr >= ind_hi)
                  {
                      ind_lo = -1;
                      break;
                  }
                  else ind_lo = ind_hi - incr;
              }
          }
      }

      // Binary Search in the estimated bracket
      while(ind_hi - ind_lo != 1)
      {
          ind_mid = ind_lo + (ind_hi - ind_lo)/2;
          if( (x >= arr[ind_mid]) == ascend)
          {
              ind_lo = ind_mid;
          }
          else
          {
              ind_hi = ind_mid;
          }
      }

      if(x == arr[n-1]) ind_lo = n-2;
      if(x == arr[0]) ind_lo = 0;
      return ind_lo;
  }

}
