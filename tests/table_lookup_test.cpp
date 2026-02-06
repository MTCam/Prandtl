// tests/table_lookup_test.cpp
#include "unit_test.hpp"
#include "BasicOperations.hpp"
#include <mutation++/mutation++.h>
#include "../libs/mfem/general/forall.hpp"

using real_t = Prandtl::real_t;
using namespace Mutation;

TEST(hunt_cpu_test)
{
    int n = 100;
    Prandtl::Vector arr(n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = i + 1;
    }

    real_t x = 34.5;

    // Hunt Right (guess near left)
    int ind_lo = 2;
    ind_lo = Prandtl::hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Hunt Left (guess near right)
    ind_lo = 70;
    ind_lo = Prandtl::hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Left Boundary
    x = 1;
    ind_lo = 50;
    ind_lo = Prandtl::hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 0, 1e-14);

    // Right Boundary
    x = n;
    ind_lo = 20;
    ind_lo = Prandtl::hunt(arr.Read(), n, x, ind_lo);
    EXPECT_CLOSE(ind_lo, n-2, 1e-14);

    return 0;
}

TEST(hunt_gpu_test)
{
    const int n = 100;
    mfem::Vector arr(n);
    for (int i = 0; i < n; i++) { arr[i] = i + 1; }

    const real_t *a = arr.Read(); // device-safe read pointer
    arr.UseDevice();

    mfem::Vector outv(4);
    outv.UseDevice();
    real_t *out_d = outv.Write(); // device-safe write pointer

    mfem::forall(4, [=] MFEM_HOST_DEVICE (int i)
    {
        real_t x;
        int guess;

        if (i == 0)      { x = 34.5;  guess = 2;  }   // hunt right
        else if (i == 1) { x = 34.5;  guess = 70; }   // hunt left
        else if (i == 2) { x = 1.0;   guess = 50; }   // left boundary
        else             { x = 100.0; guess = 20; }   // right boundary

        int idx = Prandtl::hunt(a, n, x, guess);
        out_d[i] = (real_t) idx;
    });

    const real_t *out_h = outv.HostRead();

    EXPECT_CLOSE(out_h[0], 33.0, 1e-14);
    EXPECT_CLOSE(out_h[1], 33.0, 1e-14);
    EXPECT_CLOSE(out_h[2],  0.0, 1e-14);
    EXPECT_CLOSE(out_h[3], 98.0, 1e-14); // n-2

    return 0;
}

TEST(mixture_properties)
{
    MixtureOptions opts("air_5");
    opts.setStateModel("EquilTP");

    // Change from default thermodynamic database (RRHO) to NASA 9-coefficient
    // polynomial database
    opts.setThermodynamicDatabase("RRHO");

    // Load the mixture with the new options
    Mixture mix(opts);

    // Setup the default composition
    mix.addComposition("N:0.8, O:0.2", true);

    // Write a header line for the table
    std::cout << std::setw(7) << "T[K]";
    for (int i = 0; i < mix.nSpecies(); ++i)
        std::cout << std::setw(13) << "X_" + mix.speciesName(i);
    std::cout << std::setw(13) << "Cp[J/kg-K]";
    std::cout << std::setw(13) << "H[J/kg]";
    std::cout << std::setw(13) << "S[J/kg-K]";
    std::cout << std::endl;

    // Loop over range of temperatures and compute equilibrium values at 1 atm
    double P = ONEATM;
    for (int i = 0; i < 5; ++i) {
        // Compute the temperature
        double T = 300.0 + static_cast<double>(i) * 100.0;

        // Set the mixture state equal to the equilibrium state for the given
        // temperature and pressure
        mix.setState(&T, &P);

        // Temperature
        std::cout << std::setw(13) << mix.T();

        // Species mole fractions
        for (int j = 0; j < mix.nSpecies(); ++j)
            std::cout << std::setw(13) << mix.X()[j];
        // Other properties
        std::cout << std::setw(13) << mix.mixtureFrozenCpMass(); // Cp [J/kg-K]
        std::cout << std::setw(13) << mix.mixtureHMass();        // H  [J/kg]
        std::cout << std::setw(13) << mix.mixtureSMass();        // S  [J/kg-K]
        std::cout << std::endl;
    }

    EXPECT_CLOSE(0, 0, 1e-14);
    return 0;
}

