#include "LaxFriedrichsFlux.hpp"

namespace Prandtl
{

real_t LaxFriedrichsFlux::Eval(const Vector &state1, const Vector &state2,
                         const Vector &nor, FaceElementTransformations &Tr,
                         Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = this->ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = this->ComputeFluxDotN(state2, nor, Tr, fluxN2);
   // NOTE: nor in general is not a unit normal
   const real_t maxE = std::max(speed1, speed2);
   // here, std::sqrt(nor*nor) is multiplied to match the scale with fluxN
   const real_t scaledMaxE = maxE*std::sqrt(nor*nor);
   for (int i=0; i<state1.Size(); i++)
   {
      flux[i] = 0.5*(scaledMaxE*(state1[i] - state2[i]) + (fluxN1[i] + fluxN2[i]));
   }
   return std::max(speed1, speed2);
}

  real_t LaxFriedrichsFlux::ComputeVolumeFlux(const Vector &state1, const Vector &state2,
                                              const Vector &metric1, const Vector &metric2,
                                              Vector &F_tilde) const
  {
    return ComputeVolumeFluxKernel(gasModel, state1.GetData(), state2.GetData(),
                                   metric1.GetData(), metric2.GetData(),
                                   F_tilde.GetData());
  }


  real_t LaxFriedrichsFlux::ComputeFaceFlux(const Vector &state1, const Vector &state2,
                                            const Vector &nor, Vector &flux) const
  {
    return ComputeFaceFluxKernel(gasModel, state1.GetData(), state2.GetData(),
                                 nor.GetData(), flux.GetData());
  }
  
}
