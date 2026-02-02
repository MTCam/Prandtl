#pragma once
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
}
