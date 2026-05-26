#pragma once

#include "LiftingScheme.hpp"

namespace Prandtl
{

using namespace mfem;

class LiftingBR1 : public LiftingScheme
{
public:
    LiftingBR1();

    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u, Vector &el_dudx, Vector &el_dudy, Vector &el_dudz) override;
    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u, Vector &el_dudx, Vector &el_dudy) override;
    virtual void AssembleLiftingFaceVector(const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr,
                                           const Vector &el_u, Vector &el_dudx) override;

    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx,
                                              Vector &el_dudy, Vector &el_dudz) override;
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr,
                                              const Vector &el_u, Vector &el_dudx, Vector &el_dudy) override;
    virtual void AssembleLiftingElementVector(const FiniteElement &el, ElementTransformation &Tr, const Vector &el_u, Vector &el_dudx) override;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables,
                                              Vector &el_dudx, Vector &el_dudy, Vector &el_dudz) override;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables,
                                              Vector &el_dudx, Vector &el_dudy) override;
    virtual void AssembleLiftingBdrFaceVector(BdrFaceIntegrator *bfi, const FiniteElement &el1, const FiniteElement &el2,
                                              FaceElementTransformations &Tr, const Vector &el_u, const ThermoTablesView &thermoTables,
                                              Vector &el_dudx) override;

};
  
}
