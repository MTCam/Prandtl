#pragma once

#include "mfem.hpp"

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
    void Conserv2Prim(const Vector &state, Vector &prim_state, real_t gammaM1);
    void Prim2Conserv(const Vector &state, Vector &conserv_state, real_t gammaM1Inverse);

    inline void Normalize(Vector &vec)
    {
        vec /= vec.Norml2();
    }

    void Cross(const Vector &vec1, const Vector &vec2, Vector &cross);
    void Normal(const Vector &vec, Vector &nor);
    void RotateState(Vector &state, const Vector &nor);
    void RotateBack(Vector &state, const Vector &nor);

    Vector ComputeRoeAverage(const Vector &state1, const Vector &state2, const real_t gamma);

    const Table& ElementIndextoBdrElementIndex(Mesh &mesh);

    
}
