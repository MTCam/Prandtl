// tests/table_lookup_test.cpp
#include "unit_test.hpp"
#include "BasicOperations.hpp"
#include <mutation++/mutation++.h>
//#include <mutation++.h>

using real_t = Prandtl::real_t;
using namespace Mutation;

TEST(hunt_test)
{
    int n = 100;
    Prandtl::Vector arr(n);
    for (int i = 0; i < n ; i++)
    {
        arr[i] = i+1;
    }
    
    real_t x = 34.5;
    int ind_lo = -1;

    // Hunt Right
    ind_lo = 2;
    Prandtl::hunt(arr, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Hunt Left
    ind_lo = 70;
    Prandtl::hunt(arr, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 33, 1e-14);

    // Left Boundary
    x = 1;
    ind_lo = 50;
    Prandtl::hunt(arr, x, ind_lo);
    EXPECT_CLOSE(ind_lo, 0, 1e-14);

    // Right Boundary
    x = n;
    ind_lo = 20;
    Prandtl::hunt(arr, x, ind_lo);
    EXPECT_CLOSE(ind_lo, n-2, 1e-14);

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

