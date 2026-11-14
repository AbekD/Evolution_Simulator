import os
import tempfile

import numpy as np
import pandas as pd

from genetics import create_human
from population import create_population
from evolution import calculate_fitness, create_next_generation
from analytics import allele_frequency, trait_frequency, save_results
from drift import (
    run_drift_experiments,
    theoretical_allele_variance,
    theoretical_mean_trait,
    variance_comparison
)


def raises_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_default_allele_frequency():
    np.random.seed(1)
    population = create_population(10000)
    assert abs(allele_frequency(population) - 50) < 2


def test_custom_allele_frequency():
    np.random.seed(1)
    population = create_population(10000, {"A": 0.7, "a": 0.3})
    assert abs(allele_frequency(population) - 30) < 2
    assert abs(trait_frequency(population) - 9) < 2      # q^2 = 0.09


def test_invalid_allele_probs():
    assert raises_value_error(create_human, allele_probs={"A": 0.6, "a": 0.6})
    assert raises_value_error(create_human, allele_probs={"A": -0.1, "a": 1.1})
    assert raises_value_error(create_human, allele_probs={"A": 1.0})
    assert raises_value_error(create_population, 0)


def test_invalid_drift_params():
    assert raises_value_error(run_drift_experiments, sizes=[])
    assert raises_value_error(run_drift_experiments, sizes=[1])
    assert raises_value_error(run_drift_experiments, sizes=[10], experiments=0)
    assert raises_value_error(run_drift_experiments, sizes=[10], generations=0)
    assert raises_value_error(run_drift_experiments, sizes=[10], allele_probs={"A": 0.9, "a": 0.9})


def test_population_size_constant():
    np.random.seed(1)
    population = create_population(37)
    for _ in range(5):
        population = create_next_generation(population, "genetic_drift")
        assert len(population) == 37


def test_drift_fitness_equals_neutral():
    human = {"has_large_hand": True}
    assert calculate_fitness(human, "genetic_drift") == calculate_fitness(human, "neutral")


def test_csv_structure():
    df = run_drift_experiments(sizes=[10, 20], experiments=2, generations=5, seed=1)
    assert list(df.columns) == ["experiment", "population_size", "generation",
                                "trait_frequency", "allele_frequency_a"]
    assert len(df) == 2 * 2 * 6                       # sizes * experiments * (5 + generation 0)
    assert df["generation"].min() == 0 and df["generation"].max() == 5
    assert df["trait_frequency"].between(0, 100).all()
    assert df["allele_frequency_a"].between(0, 100).all()

    path = os.path.join(tempfile.gettempdir(), "drift_test.csv")
    save_results(df, path)
    assert list(pd.read_csv(path).columns) == list(df.columns)
    os.remove(path)


def test_seed_reproducible():
    first = run_drift_experiments(sizes=[10, 20], experiments=2, generations=5, seed=42)
    np.random.rand(100)                                # disturb the global generator
    second = run_drift_experiments(sizes=[10, 20], experiments=2, generations=5, seed=42)
    other = run_drift_experiments(sizes=[10, 20], experiments=2, generations=5, seed=43)

    assert first.equals(second)
    assert not first.equals(other)


def test_fixation_is_permanent():
    population = [create_human(["a", "a"]) for _ in range(20)]
    for _ in range(5):
        population = create_next_generation(population, "genetic_drift")
        assert allele_frequency(population) == 100
        assert trait_frequency(population) == 100


def test_theory_formulas():
    # generation 0 is one sampling step: Var = q(1-q)/(2N)
    assert np.isclose(theoretical_allele_variance(0.5, 50, 0), 0.25 / 100)
    # without drift the trait frequency is just q^2
    assert np.isclose(theoretical_mean_trait(0.5, 50, 0), 25)
    # variance grows with time and is larger for small populations
    assert theoretical_allele_variance(0.5, 50, 20) > theoretical_allele_variance(0.5, 50, 5)
    assert theoretical_allele_variance(0.5, 50, 20) > theoretical_allele_variance(0.5, 500, 20)


def test_simulation_matches_theory():
    # Many short experiments with a small population -> fast and statistically clear
    df = run_drift_experiments(sizes=[20], experiments=300, generations=10, seed=7)
    last = df[df["generation"] == 10]

    mean_allele = last["allele_frequency_a"].mean()
    assert abs(mean_allele - 50) < 2.5, mean_allele   # drift does not push the mean

    var_df = variance_comparison(df)
    row = var_df[var_df["generation"] == 10].iloc[0]
    ratio = row["simulated_variance"] / row["theoretical_variance"]
    assert 0.75 < ratio < 1.25, ratio                 # sampling noise of ~8% with 300 runs

    mean_trait = last["trait_frequency"].mean()
    expected = theoretical_mean_trait(0.5, 20, 10)
    assert abs(mean_trait - expected) < 2.5, (mean_trait, expected)
    assert mean_trait > 25                            # above q0^2 = 25%


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASSED", test.__name__)
    print(f"\nAll {len(tests)} tests passed")