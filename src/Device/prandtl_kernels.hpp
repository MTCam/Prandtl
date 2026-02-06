#pragma once
#include <cmath>
// Drag in essential parts of MFEM for kernels
#include "config/config.hpp"
#include "general/forall.hpp"
#ifndef MFEM_HOST_DEVICE
#include "general/device.hpp"
#endif
#ifndef MFEM_HOST_DEVICE
#error "MFEM_HOST_DEVICE not defined. Check MFEM headers/includes."
#endif

namespace Prandtl
{

#ifdef MFEM_USE_SINGLE
  using real_t = float;
#else
  using real_t = double;
#endif

  namespace Kernels {
    MFEM_HOST_DEVICE inline real_t rmax(real_t a, real_t b) { return a > b ? a : b; }
    MFEM_HOST_DEVICE inline real_t rsqrt(real_t x) { return std::sqrt(x); }  // mfem::sqrt?
    MFEM_HOST_DEVICE inline real_t rlog(real_t x)  { return std::log(x); }   // mfe::log?
    
    MFEM_HOST_DEVICE
    inline void ComputeMeanVec(const real_t* a, const real_t* b, real_t* out, int n)
    {
      for (int i=0;i<n;++i) out[i] = real_t(0.5)*(a[i]+b[i]);
    }
    
    MFEM_HOST_DEVICE
    inline real_t ComputeLogMean(real_t x, real_t y, real_t eps) // eps defaults to 1e-4 on CPU
    {
      const real_t xi = y / x;
      const real_t u  = (xi*(xi - 2.0) + 1.0) / (xi*(xi + 2.0) + 1.0);
      
      // polynomial approximation branch when u is small
      if (u < eps)
        {
          // (x+y)*52.5 / (105 + u*(35 + u*(21 + 15*u)))
          const real_t denom = 105.0 + u*(35.0 + u*(21.0 + 15.0*u));
          return (x + y) * 52.5 / denom;
        }
      else
        {
          return (y - x) / Kernels::rlog(xi);
        }
    }
  }
}
