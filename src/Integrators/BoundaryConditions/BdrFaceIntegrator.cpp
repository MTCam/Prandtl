#include "BdrFaceIntegrator.hpp"
#include "LiftingScheme.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{

constexpr bool debug_boundary = false;

  BdrFaceIntegrator::BdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme_, const ActiveGasModel &gasModel_,
                                       const NumericalFlux &rsolver, int Np, const real_t &time, bool constant, bool t_dependent)
    : NonlinearFormIntegrator(), gasModel(gasModel_),
      liftingScheme(liftingScheme_),
      rsolver(rsolver), fluxFunction(rsolver.GetFluxFunction()),
      Np_x(Np), GLIntRules(0, Quadrature1D::GaussLobatto), 
      num_equations(fluxFunction.num_equations), dim(fluxFunction.dim),
      mass_eq(gasModel.L.eq_mass), mom_eq(gasModel.L.eq_mom[0]),
      en_eq(gasModel.L.eq_energy), sc_eq(gasModel.L.eq_scalar0),
      time(time), constant(constant), t_dependent(t_dependent)
{
   const IntegrationRule *ir_vol;

    IntegrationOrder = 2 * Np_x - 3;

   ir = &GLIntRules.Get(Geometry::SEGMENT, IntegrationOrder);
   if (dim == 1)
   {
      ir_face = &GLIntRules.Get(Geometry::POINT, IntegrationOrder);
      ir_vol = &GLIntRules.Get(Geometry::SEGMENT,IntegrationOrder);
   }
   else if (dim == 2)
   {
      ir_face = &GLIntRules.Get(Geometry::SEGMENT, IntegrationOrder);
      ir_vol = &GLIntRules.Get(Geometry::SQUARE,IntegrationOrder);
   }
   else
   {
      ir_face = &GLIntRules.Get(Geometry::SQUARE, IntegrationOrder);
      ir_vol = &GLIntRules.Get(Geometry::CUBE,IntegrationOrder);
   }

   max_char_speed = -1.0;

   dof1 = ir_vol->GetNPoints();
   shape1.SetSize(ir_vol->GetNPoints());

   state1.SetSize(num_equations);
   state2.SetSize(num_equations);
   conserv_state.SetSize(num_equations);

   dqdx.SetSize(num_equations);
   dqdy.SetSize(num_equations);
   dqdz.SetSize(num_equations);

   flux_num.SetSize(num_equations);
   flux_mat.SetSize(num_equations, dim);

   flux_mat = 0.0;

   nor.SetSize(dim);

   dU_face1.SetSize(num_equations);
}

void BdrFaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                           FaceElementTransformations &Tr,
                                           const Vector &el_u,
                                           Vector &el_dudt)
{
   if (debug_boundary) // && (time >= 2.99999 || time == 0))
   {
      std::cout << "===== Entering BdrFaceIntegrator::AssembleFaceVector =====" << std::endl;
   }

   el_dudt.SetSize(dof1 * num_equations);
   el_dudt = 0.0;

   const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);

   DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);
   
   for (int i = 0; i < ir_face->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_face->IntPoint(i);
      Tr.SetAllIntPoints(&ip); // set face and element int. points
      J1 = Tr.GetElement1Transformation().Weight();
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

      Vector phys(dim);
      Tr.Transform(ip, phys);
      real_t r = phys[1]; 

      el_u_mat1.MultTranspose(shape1, state1);

      if (dim == 1)
      {
         nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      ThermoTablesView thermoTables; // CL ALERT : Placeholder for now, can't get it as an input because this is a virtual function of NonlinearFormIntegrator.
      max_char_speed = std::max(max_char_speed, ComputeBdrFaceInviscidFlux(state1, state2, dU_face1, nor, Tr, ip, thermoTables));

        if (debug_boundary)
        {
            std::cout << "[BdrFace]" << " Elem      =  " << Tr.ElementNo << "\n";
            std::cout << "[BdrFace]" << " (x,r)     = (" << phys[0] << ", " << std::round(r) << ")" << "\n";
            std::cout << "[BdrFace]" << " faceIP#   =  " << i <<", ip.x = " << ip.x << ", weight = " << ip.weight << ", J1 = " << J1 << "\n";
            std::cout << "[BdrFace]" << " state1    = [" << state1[0] << ", " << state1[1] << ", " << state1[2] << ", " << state1[3] << "]\n";
            std::cout << "[BdrFace]" << " state2    = [" << state2[0] << ", " << state2[1] << ", " << state2[2] << ", " << state2[3] << "]\n";
            std::cout << "[BdrFace]" << " dU_face1  = [" << dU_face1[0] << ", " << dU_face1[1] << ", " << dU_face1[2] << ", " << dU_face1[3] << "]\n";
        }

#ifdef AXISYMMETRIC
      dU_face1 *= r;
#endif

    if (debug_boundary)
        {
            std::cout << "[BdrFace]" << " rdU_face1 = [" << dU_face1[0] << ", " << dU_face1[1] << ", " << dU_face1[2] << ", " << dU_face1[3] << "]\n";
            std::cout << "________________________________________________________________________________________________________________" << "\n";
        }
      dU_face1.Neg();

      // pre-multiply integration weight to flux
      AddMult_a_VWt(1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
   }
}


void BdrFaceIntegrator::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u,
                                                  const ThermoTablesView &thermoTables,
                                                  Vector &el_dudx, Vector &el_dudy, Vector &el_dudz)
{
   liftingScheme->AssembleLiftingBdrFaceVector(this, el1, el2, Tr, el_u, thermoTables, el_dudx, el_dudy, el_dudz);
}

void BdrFaceIntegrator::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u,
                                                  const ThermoTablesView &thermoTables,
                                                  Vector &el_dudx, Vector &el_dudy)
{
   liftingScheme->AssembleLiftingBdrFaceVector(this, el1, el2, Tr, el_u, thermoTables, el_dudx, el_dudy);
}

void BdrFaceIntegrator::AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u,
                                                  const ThermoTablesView &thermoTables,
                                                  Vector &el_dudx)
{
   liftingScheme->AssembleLiftingBdrFaceVector(this, el1, el2, Tr, el_u, thermoTables, el_dudx);
}

void BdrFaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u,
                                           const Vector &el_dudx, const Vector &el_dudy, const Vector &el_dudz,
                                           const ThermoTablesView &thermoTables,
                                           Vector &el_dudt)
{
   el_dudt.SetSize(dof1 * num_equations);
   el_dudt = 0.0;

   const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
   const DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
   const DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
   const DenseMatrix el_dudz_mat1(el_dudz.GetData(), dof1, num_equations);
   DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);

   for (int i = 0; i < ir_face->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_face->IntPoint(i);
      Tr.SetAllIntPoints(&ip);
      J1 = Tr.GetElement1Transformation().Weight();
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

      el_u_mat1.MultTranspose(shape1, state1);
      el_dudx_mat1.MultTranspose(shape1, dqdx);
      el_dudy_mat1.MultTranspose(shape1, dqdy);
      el_dudz_mat1.MultTranspose(shape1, dqdz);

      CalcOrtho(Tr.Jacobian(), nor);

      max_char_speed = std::max(max_char_speed, ComputeBdrFaceInviscidFlux(state1, state2, dU_face1, nor, Tr, ip, thermoTables));
      dU_face1.Neg();

      ComputeBdrFaceViscousFlux(state1, state2, dqdx, dqdy, dqdz, flux_num, nor, Tr, ip, thermoTables);
   
      dU_face1 += flux_num;

      AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
   }   
}

void BdrFaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                                           FaceElementTransformations &Tr, const Vector &el_u,
                                           const Vector &el_dudx, const Vector &el_dudy,
                                           const ThermoTablesView &thermoTables,
                                           Vector &el_dudt)
{
   el_dudt.SetSize(dof1 * num_equations);
   el_dudt = 0.0;

   const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
   const DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
   const DenseMatrix el_dudy_mat1(el_dudy.GetData(), dof1, num_equations);
   DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);

   for (int i = 0; i < ir_face->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_face->IntPoint(i);
      Tr.SetAllIntPoints(&ip);
      J1 = Tr.GetElement1Transformation().Weight();
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

      el_u_mat1.MultTranspose(shape1, state1);
      el_dudx_mat1.MultTranspose(shape1, dqdx);
      el_dudy_mat1.MultTranspose(shape1, dqdy);

      CalcOrtho(Tr.Jacobian(), nor);

      max_char_speed = std::max(max_char_speed, ComputeBdrFaceInviscidFlux(state1, state2, dU_face1, nor, Tr, ip, thermoTables));
      dU_face1.Neg();

      ComputeBdrFaceViscousFlux(state1, state2, dqdx, dqdy, flux_num, nor, Tr, ip, thermoTables);

      dU_face1 += flux_num;

      AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), shape1, dU_face1, el_dudt_mat1);
   }   
}

void BdrFaceIntegrator::AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u,
                                           const Vector &el_dudx,
                                           const ThermoTablesView &thermoTables,
                                           Vector &el_dudt)
{
   const int dof1 = el1.GetDof();

   shape1.SetSize(dof1);

   el_dudt.SetSize(dof1 * num_equations);
   el_dudt = 0.0;

   const DenseMatrix el_u_mat1(el_u.GetData(), dof1, num_equations);
   const DenseMatrix el_dudx_mat1(el_dudx.GetData(), dof1, num_equations);
   DenseMatrix el_dudt_mat1(el_dudt.GetData(), dof1, num_equations);

   for (int i = 0; i < ir_face->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_face->IntPoint(i);
      Tr.SetAllIntPoints(&ip);
      J1 = Tr.GetElement1Transformation().Weight();
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

      el_u_mat1.MultTranspose(shape1, state1);
      el_dudx_mat1.MultTranspose(shape1, dqdx);

      nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;

      max_char_speed = std::max(max_char_speed, ComputeBdrFaceInviscidFlux(state1, state2, dU_face1, nor, Tr, ip, thermoTables));
      dU_face1.Neg();

      ComputeBdrFaceViscousFlux(state1, state2, dqdx, flux_num, nor, Tr, ip, thermoTables);
   
      dU_face1 += flux_num;

      AddMult_a_VWt(+1.0 / (ir->IntPoint(0).weight * J1), dU_face1, dU_face1, el_dudt_mat1);
   }   
}

void BdrFaceIntegrator::ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr,
                                                  const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
   mfem_error("BdrFaceIntegrator::ComputeOuterInviscidState() is not overridden!");
}

real_t BdrFaceIntegrator::ComputeBdrFaceInviscidFlux(const Vector &state1, Vector &state2, Vector &fluxN, const Vector &nor,
                                                     FaceElementTransformations &Tr, const IntegrationPoint &ip,
                                                     const ThermoTablesView &thermoTables)
{
   ComputeOuterInviscidState(state1, state2, Tr, ip, thermoTables);
   // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
   // Compute hat(F) using evaluated quantities
   return rsolver.ComputeFaceFlux(state1, state2, nor, thermoTables, fluxN);
}

void BdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx,
                                                  const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor,
                                                  FaceElementTransformations &Tr, const IntegrationPoint &ip,
                                                  const ThermoTablesView &thermoTables)
{
   mfem_error("BdrFaceIntegrator::ComputeBdrFaceViscousFlux() is not overridden!");
}

void BdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx,
                                                  const Vector &dqdy, Vector &fluxN, const Vector &nor,
                                                  FaceElementTransformations &Tr, const IntegrationPoint &ip,
                                                  const ThermoTablesView &thermoTables)
{
   mfem_error("BdrFaceIntegrator::ComputeBdrFaceViscousFlux() is not overridden!");
}

void BdrFaceIntegrator::ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx,
                                                  Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr,
                                                  const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
{
   mfem_error("BdrFaceIntegrator::ComputeBdrFaceViscousFlux() is not overridden!");
}

  void BdrFaceIntegrator::ComputeBdrFaceLiftingFlux(const Vector &state1, Vector &fluxN, FaceElementTransformations &Tr,
                                                    const IntegrationPoint &ip, const ThermoTablesView &thermoTables)
  {
    Entropy2Conserv(gasModel, thermoTables, state1, conserv_state);
    ComputeOuterInviscidState(conserv_state, state2, Tr, ip, thermoTables);
    Conserv2Entropy(gasModel, thermoTables, state2, fluxN);
    fluxN -= state1;
    fluxN *= 0.5; 
  }
  
}
