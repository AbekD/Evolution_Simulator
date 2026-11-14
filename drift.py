import sys

import numpy as np
import pandas as pd

from genetics import validate_allele_probs
from population import create_population
from evolution import create_next_generation
from analytics import (
    trait_frequency,
    allele_frequency,
    plot_drift_trajectories,
    plot_drift_mean_comparison,
    plot_variance_comparison,
    save_results
)

DEFAULT_SIZES = [50, 100, 500, 1000]


def validate_drift_params(sizes, experiments, generations):
    if len(sizes) == 0:
        raise ValueError("sizes must contain at least one population size")

    for size in sizes:
        if not isinstance(size, int) or size < 2:
            raise ValueError("Each population size must be an integer >= 2")

    if not isinstance(experiments, int) or experiments < 1:
        raise ValueError("experiments must be a positive integer")

    if not isinstance(generations, int) or generations < 1:
        raise ValueError("generations must be a positive integer")


def run_single_drift(size, generations, allele_probs=None):
    population = create_population(size, allele_probs)

    rows = [{
        "generation": 0,
        "trait_frequency": trait_frequency(population),
        "allele_frequency_a": allele_frequency(population)
    }]

    for generation in range(1, generations + 1):
        population = create_next_generation(population, "genetic_drift")

        rows.append({
            "generation": generation,
            "trait_frequency": trait_frequency(population),
            "allele_frequency_a": allele_frequency(population)
        })

    return rows


def run_drift_experiments(sizes=None, experiments=30, generations=100,
                          allele_probs=None, seed=None):
    if sizes is None:
        sizes = DEFAULT_SIZES

    validate_drift_params(sizes, experiments, generations)

    if allele_probs is not None:
        validate_allele_probs(allele_probs)

    if seed is not None:
        np.random.seed(seed)

    data = []
    for size in sizes:
        for experiment in range(1, experiments + 1):
            for row in run_single_drift(size, generations, allele_probs):
                data.append({
                    "experiment": experiment,
                    "population_size": size,
                    "generation": row["generation"],
                    "trait_frequency": row["trait_frequency"],
                    "allele_frequency_a": row["allele_frequency_a"]
                })

    return pd.DataFrame(data)


def get_start_frequency(allele_probs):
    if allele_probs is None:
        return 0.5

    return allele_probs["a"]


def theoretical_allele_variance(q0, size, generation):
    # Wright-Fisher: the population has 2N alleles, each new generation is a
    # random sample of 2N alleles from the previous one.
    # Var(q_t) = q0 * (1 - q0) * (1 - (1 - 1/(2N))^t)
    # The initial population (generation 0) is already one random sample
    # of 2N alleles, so generation g has g + 1 sampling steps.
    steps = generation + 1

    return q0 * (1 - q0) * (1 - (1 - 1 / (2 * size)) ** steps)


def theoretical_mean_trait(q0, size, generation):
    # Expected % of 'aa' individuals.
    if generation == 0:
        extra = 0
    else:
        extra = theoretical_allele_variance(q0, size, generation - 1)

    return (q0 ** 2 + extra) * 100


def variance_comparison(df, allele_probs=None):
    q0 = get_start_frequency(allele_probs)
    rows = []

    for size in sorted(df["population_size"].unique()):
        subset = df[df["population_size"] == size]
        simulated = (subset["allele_frequency_a"] / 100).groupby(subset["generation"]).var()

        for generation, variance in simulated.items():
            rows.append({
                "population_size": size,
                "generation": generation,
                "simulated_variance": variance,
                "theoretical_variance": theoretical_allele_variance(q0, size, generation)
            })

    return pd.DataFrame(rows)


def summarize_drift(df, allele_probs=None):
    q0 = get_start_frequency(allele_probs)
    last_generation = df["generation"].max()
    var_df = variance_comparison(df, allele_probs)
    rows = []

    for size in sorted(df["population_size"].unique()):
        final = df[(df["population_size"] == size) & (df["generation"] == last_generation)]
        var_row = var_df[(var_df["population_size"] == size) &
                         (var_df["generation"] == last_generation)].iloc[0]

        rows.append({
            "population_size": size,
            "experiments": len(final),
            "mean_trait_%": final["trait_frequency"].mean(),
            "theory_trait_%": theoretical_mean_trait(q0, size, last_generation),
            "std_trait_%": final["trait_frequency"].std(),
            "mean_allele_%": final["allele_frequency_a"].mean(),
            "var_allele_sim": var_row["simulated_variance"],
            "var_allele_theory": var_row["theoretical_variance"],
            "a_fixed": int((final["allele_frequency_a"] == 100).sum()),
            "a_lost": int((final["allele_frequency_a"] == 0).sum())
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    experiments = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = run_drift_experiments(experiments=experiments, seed=seed)
    save_results(df, "data/genetic_drift_results.csv")
    print(df.head())
    print("Saved", len(df), "rows to data/genetic_drift_results.csv")

    print()
    print(summarize_drift(df).round(4).to_string(index=False))

    plot_drift_trajectories(df)
    plot_drift_mean_comparison(df)
    plot_variance_comparison(variance_comparison(df))