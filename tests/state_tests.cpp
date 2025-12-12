// tests/state_tests.cpp
#include "unit_test.hpp"
#include "state_semantics.hpp"
#include "legacy_state_adapter.hpp"

TEST(LegacyState_MassMomentumEnergy)
{
    const int dim   = 2;   // or test dim=3 in another test
    const int ndofs = 5;   // small number is fine

    LegacyConservativeState state(dim, ndofs);
    run_basic_mass_momentum_energy_test(state);

    return 0;
}
