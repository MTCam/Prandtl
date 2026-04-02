#include "BasicOperations.hpp"

namespace Prandtl
{

real_t ComputeLogMean(real_t x, real_t y, real_t eps)
{
    real_t xi = y / x;
    real_t u = (xi * (xi - 2.0) + 1.0) / (xi * (xi + 2.0) + 1);
    real_t ln_mean = (u < eps) ? (x + y) * 52.5 / (105.0 + u * (35.0 + u * (21.0 + 15.0 * u))) : (y - x) / std::log(xi);
    return ln_mean;
}

void AddRow(DenseMatrix &A, const Vector &row, int r)
{
    MFEM_ASSERT(A.Width() == row.Size(), "");
    MFEM_ASSERT(row.GetData() != nullptr, "supplied row pointer is null");
    for (int j = 0; j < A.Width(); j++)
    {
        A(r, j) += row[j];
    }
}

void ComputeMean(const Vector &x, const Vector &y, Vector &mean)
{
    mean = x;
    mean += y;
    mean *= 0.5;
}

real_t ComputePressure(const Vector &state, real_t gammaM1)
{
  Vector mom(state.GetData() + 1, state.Size() - 2);
  return gammaM1 * (state(state.Size() - 1) - 0.5 * (mom * mom) / state(0));
}

real_t ComputeEntropy(real_t rho, real_t p, real_t gamma)
{
#ifdef PARABOLIC
    return 0; // update this for FR (for NS) in the future 
#else
    return rho > 0.0 && p > 0.0 ? rho * log(p * pow(rho, -gamma)) : infinity();
#endif
}

real_t ComputeInternalEnergy(real_t p, real_t rho, real_t gammaM1Inverse, real_t b)
{
    return p * (1.0 - b * rho) / rho * gammaM1Inverse;
}

real_t ComputeSoundSpeed(real_t p, real_t rho, real_t gamma, real_t b)
{
    return std::sqrt(gamma * p / (1.0 - b * rho) / rho);
}

real_t ComputeEnthalpy(real_t p, real_t rho, real_t e)
{
    return e + p / rho;
}

real_t ComputeTotalEnthalpy(const Vector &state, real_t gammaM1)
{
    return (state(state.Size() - 1) + ComputePressure(state, gammaM1)) / state(0);
}

void Conserv2Entropy(const DenseMatrix &vdof_mat, DenseMatrix &ent_mat, real_t gamma, real_t gammaM1, real_t gammaM1Inverse)
{
    ent_mat = 0.0;
    Vector state, ent_state(vdof_mat.Width());
    for (int d = 0; d < vdof_mat.Height(); d++)
    {
        vdof_mat.GetRow(d, state);
        Conserv2Entropy(state, ent_state, gamma, gammaM1, gammaM1Inverse);
        ent_mat.SetRow(d, ent_state);
    }
}

void Conserv2Entropy(const Vector &state, Vector &ent_state, real_t gamma, real_t gammaM1, real_t gammaM1Inverse)
{
    real_t s, p, v_sq, beta;

    Vector mom(state.GetData() + 1, state.Size() - 2);
    const real_t rho = state(0);
    const real_t rhom1 = 1.0 / rho;
    v_sq = 0.5 * (mom * mom) * rhom1 * rhom1;

    p = gammaM1 * (state(state.Size() - 1) - rho * v_sq);
    beta = rho / p;
    s = std::log(p * std::pow(rho, -gamma));

    ent_state(0) = (gamma - s) * gammaM1Inverse - beta * v_sq;
    ent_state(1) = beta * mom(0) * rhom1;
    if (state.Size() > 3)
    {
        ent_state(2) = beta * mom(1) * rhom1;
        if (state.Size() > 4)
        {
            ent_state(3) = beta * mom(2) * rhom1;
        }
    }
    ent_state(state.Size() - 1) = -beta;
}

void Entropy2Conserv(const Vector &ent_state, Vector &state, real_t gamma, real_t gammaM1, real_t gammaM1Inverse)
{
    int dim = ent_state.Size() - 2;
    Vector vel(dim);
    const real_t beta = -ent_state(dim + 1);
    for(int i = 0;i < dim;i++) vel(i) = ent_state(i+1) / beta;
    const real_t s = gamma - (ent_state(0) + 0.5 * beta * (vel * vel)) * gammaM1;
    state(0) = std::pow(std::exp(-s) / beta, gammaM1Inverse);
    state(1) = state(0) * vel(0);
    if (dim > 1)
    {
        state(2) = state(0) * vel(1);
        if (dim > 2)
        {
            state(3) = state(0) * vel(2);
        }
    }
    state(dim + 1) = state(0) * (1.0 / beta * gammaM1Inverse + 0.5 * (vel * vel));
}

void EntropyGrad2PrimGrad(const DenseMatrix &vdof_mat, DenseMatrix &grad, real_t gammaM1, real_t gammaM1Inverse)
{
    Vector state, grad_state;
    real_t KE, p, v_sq;
    int dim = vdof_mat.Width() - 2;

    for (int d = 0; d < vdof_mat.Height(); d++)
    {
        vdof_mat.GetRow(d, state);
        grad.GetRow(d, grad_state);

        Vector vel(dim);
        for(int i = 0;i < dim;i++) vel(i) = state(i+1)/state(0);
        v_sq = 0.5 * (vel * vel);
        p = gammaM1 * (state(state.Size() - 1) - state(0) * v_sq);
        KE = v_sq * state(0);

        if (dim == 1)
        {
            grad(d, 1) = p / state(0) * (grad_state(1) + vel(0) * grad_state(2));
            grad(d, 0) = state(0) * grad_state(0) - grad_state(2) * (KE - p * gammaM1Inverse)
                       + state(0) / p * state(1) * grad(d, 1);
            grad(d, 2) = p / state(0) * (grad(d, 0) + p * grad_state(2));
        }
        else if (dim == 2)
        {
            grad(d, 1) = p / state(0) * (grad_state(1) + vel(0) * grad_state(3));
            grad(d, 2) = p / state(0) * (grad_state(2) + vel(1) * grad_state(3));
            grad(d, 0) = state(0) * grad_state(0) - grad_state(3) * (KE - p * gammaM1Inverse) 
                       + state(0) / p * (state(1) * grad(d, 1) + state(2) * grad(d, 2));
            grad(d, 3) = p / state(0) * (grad(d, 0) + p * grad_state(3));

        }
        else
        {
            grad(d, 1) = p / state(0) * (grad_state(1) + vel(0) * grad_state(4));
            grad(d, 2) = p / state(0) * (grad_state(2) + vel(1) * grad_state(4));
            grad(d, 3) = p / state(0) * (grad_state(3) + vel(2) * grad_state(4));
            grad(d, 0) = state(0) * grad_state(0) - grad_state(4) * (KE - p * gammaM1Inverse) 
                       + state(0) / p * (state(1) * grad(d, 1) + state(2) * grad(d, 2) +
                                         state(3) * grad(d, 3));
            grad(d, 4) = p / state(0) * (grad(d, 0) + p * grad_state(4));
        }
    }
}


void Conserv2Prim(const Vector &state, Vector &prim_state, real_t gammaM1)
{
    prim_state.SetSize(state.Size());
    prim_state(0) = state(0);
    prim_state(1) = state(1) / state(0);
    if (state.Size() > 3)
    {
        prim_state(2) = state(2) / state(0);
        if (state.Size() > 4)
        {
          prim_state(3) = state(3) / state(0);
        }
    }
    prim_state(state.Size() - 1) = ComputePressure(state, gammaM1);
}

  void Prim2Conserv(const Vector &state, Vector &conserv_state, real_t gammaM1Inverse)
{
    conserv_state.SetSize(state.Size());
    conserv_state(0) = state(0);
    conserv_state(1) = state(0) * state(1);
    if (state.Size() > 3)
    {
        conserv_state(2) = state(0) * state(2);
        if (state.Size() > 4)
        {
            conserv_state(3) = state(0) * state(3);
        }
    }
    real_t ke = 0.0;
    for (int i = 0; i < state.Size() - 2; i++)
    {
        ke += state(i + 1) * state(i + 1);
    }
    ke *= state(0);
    ke *= 0.5;
    conserv_state(state.Size() - 1) = state(state.Size() - 1) * gammaM1Inverse + ke;
}

void Cross(const Vector &vec1, const Vector &vec2, Vector &cross)
{
    MFEM_ASSERT(vec1.Size() == 3 && vec2.Size() == 3,
                "Apply cross product only to 3D vectors");
    cross.SetSize(3);

    cross(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    cross(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    cross(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);
}

void Normal(const Vector &vec, Vector &nor)
{
    MFEM_ASSERT(vec.Size() == 2 || vec.Size() == 3, "Apply only to 2D/3D vectors");
    
    nor.SetSize(vec.Size());

    nor(0) = vec.Size() == 2? -vec(1) :
        (std::abs(vec(0)) > 1e-10 ? -(vec(1) / vec(0)) / std::sqrt(1.0 + std::pow(vec(1) / vec(0), 2)) : -1.0);
    nor(1) = vec.Size() == 2? vec(0) : 
        (std::abs(vec(0)) > 1e-10 ? 1.0 / std::sqrt(1.0 + std::pow(vec(1) / vec(0), 2)) : 0.0);
    if (vec.Size() == 3)
    {
        nor(2) = 0.0;
    }
}


void RotateState(Vector &state, const Vector &nor)
{
    MFEM_ASSERT(nor.Size() > 1, "Rotate only in 2D or 3D");
    MFEM_ASSERT(nor.Size() < 4, "Rotate only in 2D or 3D");
 
    Vector tan1;
    Vector rhoV(state.GetData() + 1, nor.Size());

    Normal(nor, tan1);
    real_t rho_u = rhoV * nor;
    real_t rho_v = rhoV * tan1;

    if (nor.Size() == 3)
    {
        Vector tan2;
        Cross(nor, tan1, tan2);
        rhoV(2) = rhoV * tan2;
    }

    rhoV(0) = rho_u;
    rhoV(1) = rho_v;
}

void RotateState(const StateLayout &layout, Vector &state, const Vector &nor)
{
  int dim = nor.Size();
  MFEM_ASSERT(dim >= 1 and dim < 4, "RotateState: Invalid normal dimension");
  
  Prandtl::PointStateViewRW S{state.GetData()};
  if(nor.Size() == 1){
    S.set_momentum(layout, 0, S.momentum(layout, 0)*nor(0));
    return;
  }
  Vector tan1;
  Vector rhoV(state.GetData() + layout.eq_mom[0], dim);
  Normal(nor, tan1);
  real_t rho_u = rhoV * nor;
  real_t rho_v = rhoV * tan1;

  if (dim == 3)
    {
      Vector tan2;
      Cross(nor, tan1, tan2);
      rhoV(2) = rhoV * tan2;
    }
  rhoV(0) = rho_u;
  rhoV(1) = rho_v;
}

void RotateBack(Vector &state, const Vector &nor)
{
    MFEM_ASSERT(nor.Size() > 1, "Rotate only in 2D or 3D");
    MFEM_ASSERT(nor.Size() < 4, "Rotate only in 2D or 3D");

    Vector tan1;
    Vector rhoV(state.GetData() + 1, nor.Size());

    Normal(nor, tan1);
    real_t rho_u = rhoV(0) * nor(0) + rhoV(1) * tan1(0);
    real_t rho_v = rhoV(0) * nor(1) + rhoV(1) * tan1(1);

    if (nor.Size() == 3)
    {
        Vector tan2;
        Cross(nor, tan1, tan2);
        rho_u += rhoV(2) * tan2(0);
        rho_v += rhoV(2) * tan2(1);
        rhoV(2) = rhoV(0) * nor(2) + rhoV(1) * tan1(2) + rhoV(2) * tan2(2);
    }

    rhoV(0) = rho_u;
    rhoV(1) = rho_v;
}

void RotateBack(Vector &state, const Vector &nor, const StateLayout &layout)
{
  int dim = nor.Size();
  MFEM_ASSERT(dim >= 1 and dim < 4, "RotateBack: Invalid normal dimension");
  Vector rhoV(state.GetData() + layout.eq_mom[0], nor.Size());
  if (dim == 1){
    rhoV(0) = rhoV(0) / nor(0);
    return;
  }
  Vector tan1;
  Normal(nor, tan1);
  real_t rho_u = rhoV(0) * nor(0) + rhoV(1) * tan1(0);
  real_t rho_v = rhoV(0) * nor(1) + rhoV(1) * tan1(1);

    if (dim == 3)
    {
        Vector tan2;
        Cross(nor, tan1, tan2);
        rho_u += rhoV(2) * tan2(0);
        rho_v += rhoV(2) * tan2(1);
        rhoV(2) = rhoV(0) * nor(2) + rhoV(1) * tan1(2) + rhoV(2) * tan2(2);
    }

    rhoV(0) = rho_u;
    rhoV(1) = rho_v;
}

Vector ComputeRoeAverage(const Vector &state1, const Vector &state2, const real_t gammaM1)
{
    Vector RoeAverage(state1.Size() + 1);
    real_t V2 = 0.0;

    Vector prim1, prim2;
    Conserv2Prim(state1, prim1, gammaM1);
    Conserv2Prim(state2, prim2, gammaM1);

    const real_t &rho1 = prim1(0);
    const real_t &rho2 = prim2(0);
    RoeAverage(0) = std::sqrt(rho1 * rho2);

    const real_t &u1 = prim1(1);
    const real_t &u2 = prim2(1);
    RoeAverage(1) = (std::sqrt(rho1) * u1 + std::sqrt(rho2) * u2) / (std::sqrt(rho1) + std::sqrt(rho2));
    V2 += RoeAverage(1) * RoeAverage(1);


    if (prim1.Size() > 3)
    {
        const real_t &v1 = prim1(2);
        const real_t &v2 = prim2(2);
        RoeAverage(2) = (std::sqrt(rho1) * v1 + std::sqrt(rho2) * v2) / (std::sqrt(rho1) + std::sqrt(rho2));
        V2 += RoeAverage(2) * RoeAverage(2);
        if (prim1.Size() > 4)
        {
            const real_t &w1 = prim1(3);
            const real_t &w2 = prim2(3);
            RoeAverage(3) = (std::sqrt(rho1) * w1 + std::sqrt(rho2) * w2) / (std::sqrt(rho1) + std::sqrt(rho2));
            V2 += RoeAverage(3) * RoeAverage(3);
        }
    }


    const real_t H1 = ComputeTotalEnthalpy(state1, gammaM1);
    const real_t H2 = ComputeTotalEnthalpy(state2, gammaM1);
    RoeAverage(RoeAverage.Size() - 2) = (std::sqrt(rho1) * H1 + std::sqrt(rho2) * H2) / (std::sqrt(rho1) + std::sqrt(rho2));

    RoeAverage(RoeAverage.Size() - 1) = std::sqrt(gammaM1 * (RoeAverage(RoeAverage.Size() - 2) -  0.5 * V2));

    return RoeAverage;
}

const Table& ElementIndextoBdrElementIndex(Mesh &mesh)
{
    Array<Connection> conn;
    conn.Reserve(mesh.GetNBE());
    int e, info;
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        mesh.GetBdrElementAdjacentElement(i, e, info);
        conn.Append(Connection(i, e));
    }
    conn.Sort();
    conn.Unique();

    Table *bdr_el2el = new Table(mesh.GetNBE(), conn);
    Table *el2bdr_el = new Table();
    Transpose(*bdr_el2el, *el2bdr_el, mesh.GetNE());

    bdr_el2el->~Table();
    return *el2bdr_el;
}

}

