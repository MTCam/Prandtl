#pragma once

#include "mfem.hpp"
#include "prandtl_device.hpp"
#include "DGSEMIntegrator.hpp"
#include "BdrFaceIntegrator.hpp"
#include "../../../libs/mfem/general/forall.hpp"

namespace Prandtl
{

  // using namespace mfem;

class DGSEMNonlinearForm : public ParNonlinearForm
{
private:
    mutable Vector aux2_x, aux2_y, aux2_z;
    Array<DGSEMIntegrator*> dnfi, fnfi;
    Array<BdrFaceIntegrator*> bfnfi;
    mutable ParGridFunction GRAD_X, GRAD_Y, GRAD_Z;
public:
    DGSEMNonlinearForm(ParFiniteElementSpace *pfes);
    void MultLifting(const Vector &u, Vector &dudx, Vector &dudy, Vector &dudz) const;
    void MultLifting(const Vector &u, Vector &dudx, Vector &dudy) const;
    void MultLifting(const Vector &u, Vector &dudx) const;

    void Mult(const Vector &u, const Vector &dudx, const Vector &dudy, const Vector &dudz, Vector &dudt) const;
    void Mult(const Vector &u, const Vector &dudx, const Vector &dudy, Vector &dudt) const;
    void Mult(const Vector &u, const Vector &dudx, Vector &dudt) const;
    void Mult(const Vector &u, Vector &dudt) const;

    void AddDomainIntegrator(DGSEMIntegrator *nlfi)
    {
        dnfi.Append(nlfi);
        dnfi_marker.Append(NULL);
    }
         
    void AddInteriorFaceIntegrator(DGSEMIntegrator *nlfi)
    {
        fnfi.Append(nlfi);
    }


    void AddBdrFaceIntegrator(BdrFaceIntegrator *bfi, Array<int> &bdr_marker)
    {
        bfnfi.Append(bfi);
        bfnfi_marker.Append(&bdr_marker);
    }

  void AssembleDeviceCache(Prandtl::DGSEMCache &dgsem_device_cache);


};

}
