#include "PerssonPeraireIndicator.hpp"
#include "BasicOperations.hpp"

namespace Prandtl
{

  PerssonPeraireIndicator::PerssonPeraireIndicator(std::shared_ptr<ParFiniteElementSpace> vfes,
                                                   std::shared_ptr<ParFiniteElementSpace> fes0,
                                                   std::shared_ptr<ParGridFunction> eta,
                                                   std::shared_ptr<ModalBasis> modalBasis,
                                                   const IdealGasModel &gasModel_)
  : Indicator(vfes, fes0, eta), modalBasis(modalBasis), gasModel(gasModel_),
    ubdegs(modalBasis->GetPolyDegs())
  {
    rho_p.SetSize(ndofs);
    modes.SetSize(ndofs);
    modesM1.SetSize(ndofs);
    modesM2.SetSize(ndofs);
    ubdegs_row.SetSize(dim);
  }

void PerssonPeraireIndicator::CheckSmoothness(const Vector &x)
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
            Prandtl::PointStateView S{state.GetData()};
            rho_p(i) = gasModel.density(S) * gasModel.pressure(S);
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
