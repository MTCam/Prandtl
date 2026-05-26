#include "PerssonPeraireIndicator.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{

  PerssonPeraireIndicator::PerssonPeraireIndicator(std::shared_ptr<ParFiniteElementSpace> vfes,
                                                   std::shared_ptr<ParFiniteElementSpace> fes0,
                                                   std::shared_ptr<ParGridFunction> eta,
                                                   std::shared_ptr<ModalBasis> modalBasis,
                                                   const ActiveGasModel &gasModel_)
  : Indicator(vfes, fes0, eta), modalBasis(modalBasis), gasModel(gasModel_),
    ubdegs(modalBasis->GetPolyDegs())
  {
    rho_p.SetSize(ndofs);
    modes.SetSize(ndofs);
    modesM1.SetSize(ndofs);
    modesM2.SetSize(ndofs);
    ubdegs_row.SetSize(dim);
  }

void PerssonPeraireIndicator::CheckSmoothness(const Vector &x, const ThermoTablesView &thermoTables)
{
    for (int el = 0; el < vfes->GetNE(); el++)
    {
        vfes->GetElementVDofs(el, vdof_indices);
        fes0->GetElementDofs(el, ind_indx);
        x.GetSubVector(vdof_indices, el_vdofs);
        DenseMatrix vdof_mat(el_vdofs.GetData(), ndofs, num_equations);

        for (int i = 0; i < ndofs; i++)
        {
            vdof_mat.GetRow(i, state);
            Prandtl::PointStateView S{state.HostRead()};
            rho_p(i) = gasModel.density(S, thermoTables) * gasModel.pressure(S, thermoTables);
        }
        modalBasis->ComputeModes(rho_p, modes);
        modesM1 = modesM2 = modes;

        for (int i = 0; i < ndofs; i++)
        {
            ubdegs.GetRow(i, ubdegs_row);
            for (int j = 0; j < dim; j++)
            {
                if (ubdegs_row[j] > order - 2)
                {
                    modesM2[i] = 0.0;
                    if (ubdegs_row[j] > order - 1)
                    {
                        modesM1[i] = 0.0;
                    }
                }
            }
        }

        ind_dof(0) = 1.0 - (modesM1 * modesM1) / (modes * modes);
        ind_dof(0) = std::max(ind_dof(0), 1.0 - (modesM2 * modesM2) / (modesM1 * modesM1));
        eta->SetSubVector(ind_indx, ind_dof);
    }
}

void PerssonPeraireIndicator::CheckIndicatorSmoothness(const Vector &indicator)
{
  const real_t *indicator_h = indicator.HostRead();
    for (int el = 0; el < vfes->GetNE(); el++)
    {
      fes0->GetElementDofs(el, ind_indx);
      // MFEM_ASSERT(ind_indx.Size() == 1, "expected one scalar dof per element");
      // MFEM_ASSERT(ind_indx(0) == el, "indicator storage not flat");
      const real_t *el_indicator = indicator_h + el*ndofs; 
        for (int i = 0; i < ndofs; i++)
        {
          rho_p(i) = el_indicator[i];
        }
        modalBasis->ComputeModes(rho_p, modes);
        modesM1 = modesM2 = modes;

        for (int i = 0; i < ndofs; i++)
        {
            ubdegs.GetRow(i, ubdegs_row);
            for (int j = 0; j < dim; j++)
            {
                if (ubdegs_row[j] > order - 2)
                {
                    modesM2[i] = 0.0;
                    if (ubdegs_row[j] > order - 1)
                    {
                        modesM1[i] = 0.0;
                    }
                }
            }
        }

        ind_dof(0) = 1.0 - (modesM1 * modesM1) / (modes * modes);
        ind_dof(0) = std::max(ind_dof(0), 1.0 - (modesM2 * modesM2) / (modesM1 * modesM1));
        eta->SetSubVector(ind_indx, ind_dof);
    }
}

}
