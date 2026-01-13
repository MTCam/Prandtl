#pragma once

#include "mfem.hpp"
#include "NumericalFlux.hpp"
#include "NavierStokesFlux.hpp"

namespace Prandtl

{

using namespace mfem;

class LiftingScheme;

/**
 * @brief Abstract boundary face integrator class
 *
 */
class BdrFaceIntegrator : public NonlinearFormIntegrator
{
private:
   real_t J1;
   // The maximum characteristic speed, updated during element/face vector assembly
   real_t max_char_speed;
   int dof1;
   int IntegrationOrder;

   Vector shape1;
   Vector flux_num;
   const int Np_x;
   IntegrationRules GLIntRules;
   const IntegrationRule *ir, *ir_face;

   Vector state1, state2, conserv_state;
   Vector dU_face1;

   std::shared_ptr<ParGridFunction> r_gf;
   std::shared_ptr<ParGridFunction> u_gf;

protected:
  Vector nor;     // normal vector, @see CalcOrtho
  const NavierStokesFlux fluxFunction;
  const int num_equations, dim;
  DenseMatrix flux_mat;
  const real_t &time;
  bool constant, t_dependent;
  Vector dqdx, dqdy, dqdz;
  std::shared_ptr<LiftingScheme> liftingScheme;
  std::shared_ptr<const IdealGasModel> gasModel;
  std::shared_ptr<const StateLayout> stateLayout;
  const NumericalFlux &rsolver;  // Numerical flux that maps F(u±,x) to hat(F)
  bool scaleStateInAxisymm = true;
  const int mass_eq, mom_eq, en_eq, sc_eq; 

public:
  BdrFaceIntegrator(std::shared_ptr<LiftingScheme> liftingScheme, std::shared_ptr<const IdealGasModel> gasModel_,
                    std::shared_ptr<const StateLayout> stateLayout_, const NumericalFlux &rsolver, int Np,
                    const real_t &time, bool constant, bool t_dependent);
   
   void ResetMaxCharSpeed()
   {
      max_char_speed = 0.0;
   }

   real_t GetMaxCharSpeed()
   {
      return max_char_speed;
   }

   const NavierStokesFlux &GetFluxFunction()
   {
      return fluxFunction;
   }

   virtual void ComputeOuterInviscidState(const Vector &state1, Vector &state2, FaceElementTransformations &Tr, const IntegrationPoint &ip);
   virtual real_t ComputeBdrFaceInviscidFlux(const Vector &state1, Vector &state2, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip);
   
   virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, const Vector &dqdz, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip);
   virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, const Vector &dqdy, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip);
   virtual void ComputeBdrFaceViscousFlux(const Vector &state1, const Vector &state2, const Vector &dqdx, Vector &fluxN, const Vector &nor, FaceElementTransformations &Tr, const IntegrationPoint &ip);
   
   virtual void ComputeBdrFaceLiftingFlux(const Vector &state1, Vector &fluxN, FaceElementTransformations &Tr, const IntegrationPoint &ip);

   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, const Vector &el_dudz, Vector &el_dudt);
   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const Vector &el_dudx, const Vector &el_dudy, Vector &el_dudt);
   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, const Vector &el_dudx, Vector &el_dudt);
   void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudt) override;

   void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz);
   void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx, Vector &el_dudy);
   void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, const Vector &el_u, Vector &el_dudx);

   virtual ~BdrFaceIntegrator() = default;
};

}
