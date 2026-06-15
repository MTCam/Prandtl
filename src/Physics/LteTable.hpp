#pragma once

#include "Physics.hpp"
#include "GasState.hpp"
#include "plato_Cpp_library_interface.h"
namespace Prandtl
{

  inline void uniform_grid(int N, real_t min, real_t max, mfem::Vector &grid)
  {
    grid.SetSize(N);
    double step = (max - min) / (N - 1);
    for (int i = 0; i < N; ++i)
    {
      grid[i] = min + i * step;
    }
  }

  inline void log_grid(int N, real_t min, real_t max, mfem::Vector &grid)
  {
    grid.SetSize(N);
    double step = std::log10(max/min) / (N-1);
    for (int i = 0;i < N; ++i)
    {
      grid[i] = min * std::pow(10, i * step);
    }
  }

  inline real_t dot_product(const real_t* a, const real_t* b, int size)
  {
    real_t result = 0.0;
    for (int i = 0; i < size; ++i)
    {
      result += a[i] * b[i];
    }
    return result;
  }

  inline void fill_lte_table(const StateLayout &L, const real_t* rho_grid, const real_t* T_grid,
                             real_t* lte_table, real_t &e_min, real_t &e_max)
  {

    real_t UKB = 1.380649e-23;

    int nb_comp = plato_get_nb_comp();
    int nb_spec = plato_get_nb_species();
    int nb_temp = plato_get_nb_temp();
    double X_tol = 1e-12;

    mfem::Vector yc(nb_comp), Xc(nb_comp);
    mfem::Vector Xi(nb_spec), Xitol(nb_spec), Xip(nb_spec), Xim(nb_spec), yi(nb_spec),
                 Ri(nb_spec), ei(nb_spec), hi(nb_spec), Ji(nb_spec),
                 Di(nb_spec),Dij(nb_spec*(nb_spec + 1)/2), di(nb_spec);
    mfem::Vector temp(nb_temp), lambda_int(nb_temp);

    real_t R, P, nb, e, betaT, alpha, cv, gam, cp, c,
           mu, sigma, lambda_trh, lambda_tre, lambda_reactive;

    plato_get_Ri(Ri.GetData());

    int flag = 0;
    for(int i=0; i < L.nx; i++)
    {
      const real_t rho = rho_grid[i];
      for(int j=0; j < L.ny; j++)
      {
        yc = 1.0;
        Xc = 1.0;

        const real_t T = T_grid[j];
        temp = T;

        // Computing LTE composition
        plato_get_eq_composition_mass(&rho, &T, yc.GetData(), yi.GetData(), &flag);
        plato_mass_to_mole_fractions(yi.GetData(), Xi.GetData());

        // Number density and pressure
        R  = dot_product(Ri.GetData(), yi.GetData(), nb_spec);
        P  = rho * R * T;
        nb = P / (UKB*T);

        // Energies and enthalpy per unit-mass
        plato_get_species_energy(temp.GetData(), ei.GetData());
        for(int sp=0; sp < nb_spec; sp++) hi[sp] = ei[sp] + Ri[sp]*T;
        e = dot_product(yi.GetData(), ei.GetData(), nb_spec);

        // Derived thermodynamic properties
        // isothermal compressibility, coeff of thermal expansion, cv, cp, gamma, sound_speed at equilibrium
        betaT = plato_get_eq_isoth_comp(&P, &T, Xi.GetData());
        alpha = plato_get_eq_coeff_th_exp(&P, &T, Xi.GetData());
        cv    = plato_get_eq_cv(&rho, &T, yi.GetData());
        gam   = 1.0 + alpha*alpha*T/(rho*betaT*cv);
        cp    = gam*cv;
        c     = std::sqrt(gam*P/rho);

        // Transport properties
        // dynamic viscosity, thermal conductivity, components of thermal conductivity
        plato_get_transp_coeff_comp(&nb, Xi.GetData(), temp.GetData(), &mu, &sigma, &lambda_trh, &lambda_tre, lambda_int.GetData(), Di.GetData());

        double eps = 1e-5;
        double Tepsp1 = T*(1.0 + eps);
        double Tepsm1 = T*(1.0 - eps);
        // reactive thermal conductivity
        // (NOTE: pass mole fractions with tolerance to procedure solving Stefan-Maxwell's equations)
        plato_get_eq_composition_mole(&P, &Tepsp1, Xc.GetData(), Xip.GetData(), &flag);
        plato_get_eq_composition_mole(&P, &Tepsm1, Xc.GetData(), Xim.GetData(), &flag);
        plato_get_bin_diff_coeff(&nb, &T, &T, Xi.GetData(), Dij.GetData());
        for(int sp=0; sp < nb_spec; sp++) di[sp] = -0.5 * (Xip[sp] - Xim[sp])/(eps*T);

        double sum_Xitol = 0.0, X;
        for(int sp=0; sp < nb_spec; sp++)
        {
          X = Xi[sp] + X_tol;
          Xitol[sp] = X;
          sum_Xitol += X;
        }

        sum_Xitol = 1.0/sum_Xitol;
        for(int sp=0; sp < nb_spec; sp++) Xitol[sp] *= sum_Xitol;

        plato_get_species_diff_flux(&T, &T, &nb, Xitol.GetData(), Dij.GetData(), di.GetData(), Ji.GetData());
        lambda_reactive = dot_product(Ji.GetData(), hi.GetData(), nb_spec);

        lte_table[L.lte_property_index(L.P_idx, i, j)] = P;
        lte_table[L.lte_property_index(L.e_idx, i, j)] = e;
        lte_table[L.lte_property_index(L.cv_idx, i, j)] = cv;
        lte_table[L.lte_property_index(L.cp_idx, i, j)] = cp;
        lte_table[L.lte_property_index(L.R_eq_idx, i, j)] = R;
        lte_table[L.lte_property_index(L.gamma_eq_idx, i, j)] = gam;
        lte_table[L.lte_property_index(L.c_idx, i, j)] = c;
        lte_table[L.lte_property_index(L.mu_idx, i, j)] = mu;
        lte_table[L.lte_property_index(L.lambda_idx, i, j)] = lambda_trh + lambda_tre + lambda_int[0] + lambda_reactive;

        if(i+j == 0)
        {
          e_min = e;
          e_max = e;
        }
        else
        {
          e_min = std::min(e_min, e);
          e_max = std::max(e_max, e);
        }
      }
    }
  }

  inline void fill_inv_table(const StateLayout &L, const real_t* rho_grid, const real_t* e_grid, const real_t* T_grid, real_t* inv_table)
  {
    int nb_comp = plato_get_nb_comp();
    int nb_spec = plato_get_nb_species();

    mfem::Vector yc(nb_comp);
    mfem::Vector yi(nb_spec), ei(nb_spec);
    mfem::Vector temp(1);

    real_t rho, e0, e, res, cv;
    real_t tol = 1e-8;

    int flag = 0;
    real_t T = T_grid[0];
    for(int i = 0; i < L.nx; i++)
    {
      rho = rho_grid[i];
      T = T_grid[0];
      for(int j = 0; j < L.ny; j++)
      {
        e0  = e_grid[j];
        res = 1.0;
        yc = 1.0;
        int it = 0;
        // Newton iteration to find T such that internal energy at (rho, T) matches e0
        while(res > tol)
        {
          temp = T;
          plato_get_eq_composition_mass(&rho, &T, yc.GetData(), yi.GetData(), &flag);
          plato_get_species_energy(temp.GetData(), ei.GetData());
          e = dot_product(yi.GetData(), ei.GetData(), nb_spec);
          cv = plato_get_eq_cv(&rho, &T, yi.GetData());

          res = -(e - e0)/cv;
          T += res;
          it++;
          res = std::abs(res)/T;

          if(it > 100)
          {
            MFEM_ABORT("Maximum number of iterations reached in fill_inv_table");
          }
          if(T<0.0)
          {
            MFEM_ABORT("Negative temperature encountered in fill_inv_table");
          }
        }
        inv_table[L.lte_property_index(0 , i, j)] = T;
      }
    }
  }
}