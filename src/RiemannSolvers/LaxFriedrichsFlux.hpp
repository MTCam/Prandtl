#pragma once

#include "mfem.hpp"

namespace Prandtl
{

using namespace mfem;


class LaxFriedrichsFlux : public FluxFunction
{
public:
   LaxFriedrichsFlux(const FluxFunction &fluxFunction)
     : FluxFunction(fluxFunction)
   {
#ifndef MFEM_THREAD_SAFE
      fluxN1.SetSize(num_equations);
      fluxN2.SetSize(num_equations);
#endif
   }

   /**
    * @brief  hat(F)n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    *
    * @param[in] state1 state value at a point from the first element
    * (num_equations)
    * @param[in] state2 state value at a point from the second element
    * (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ(u⁺ - u⁻)
    */
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const;

protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1, fluxN2;
#endif
};


}
