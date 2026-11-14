import pandas as pd
import matplotlib.pyplot as plt

from genetics import reproduce

def trait_count(population):
  return sum(person['has_large_hand'] for person in population)


def trait_frequency(population):
  big = trait_count(population)

  return (big / len(population)) * 100


def allele_frequency(population):
  # Percentage of 'a' alleles among all 2N alleles in the population
  a_count = sum(person['genotype'].count('a') for person in population)

  return (a_count * 100) / (2 * len(population))


def test_inheritance(parent1, parent2, children_count):
  children = []

  for _ in range(children_count):
    children.append(reproduce(parent1, parent2))

  big_arms = trait_count(children)
  normal_arms = len(children) - big_arms

  return big_arms, normal_arms


def plot_results(df):
    groups = df.groupby(
    ["generation", "scenario"]
    )["frequency"].mean().reset_index()
    groups = groups.sort_values("generation")

    result = groups.pivot(
        index="generation",
        columns="scenario",
        values="frequency"
    )
    result = result.sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(result.index, result["neutral"], label="Neutral")
    plt.plot(result.index, result["positive"], label="Positive")
    plt.plot(result.index, result["negative"], label="Negative")
    plt.xlabel("Generation")
    plt.ylabel("Trait frequency (%)")
    plt.title("Change in Trait Frequency Across Generations")
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.show()


def save_results(df, filename):
    df.to_csv(filename, index=False)


def plot_drift_trajectories(df, column="trait_frequency", save_path=None):
    labels = {
        "trait_frequency": "Trait frequency (%)",
        "allele_frequency_a": "Allele a frequency (%)"
    }
    sizes = sorted(df["population_size"].unique())

    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, size in zip(axes, sizes):
        subset = df[df["population_size"] == size]

        for experiment, run in subset.groupby("experiment"):
            ax.plot(run["generation"], run[column],
                    color="gray", alpha=0.4, linewidth=0.8)

        mean = subset.groupby("generation")[column].mean()
        ax.plot(mean.index, mean.values, color="red", linewidth=2, label="Mean")

        ax.set_title(f"N = {size}")
        ax.set_xlabel("Generation")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(labels.get(column, column))
    axes[0].legend()
    fig.suptitle("Genetic Drift: " + labels.get(column, column))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def plot_drift_mean_comparison(df, show_band=True, save_path=None):
    plt.figure(figsize=(10, 6))

    for size in sorted(df["population_size"].unique()):
        subset = df[df["population_size"] == size]
        grouped = subset.groupby("generation")["trait_frequency"]
        mean = grouped.mean()
        std = grouped.std()

        plt.plot(mean.index, mean.values, label=f"N = {size}")

        if show_band:
            lower = (mean - std).clip(lower=0)
            upper = (mean + std).clip(upper=100)
            plt.fill_between(mean.index, lower, upper, alpha=0.15)

    plt.xlabel("Generation")
    plt.ylabel("Trait frequency (%)")
    plt.title("Mean Trait Frequency by Population Size (band = +-1 std between experiments)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def plot_variance_comparison(var_df, save_path=None):
    plt.figure(figsize=(10, 6))

    for size in sorted(var_df["population_size"].unique()):
        subset = var_df[var_df["population_size"] == size]

        line, = plt.plot(subset["generation"], subset["simulated_variance"],
                         label=f"N = {size} (simulated)")
        plt.plot(subset["generation"], subset["theoretical_variance"],
                 linestyle="--", color=line.get_color(),
                 label=f"N = {size} (theory)")

    plt.xlabel("Generation")
    plt.ylabel("Variance of allele frequency of a")
    plt.title("Allele Frequency Variance: Simulation vs Wright-Fisher Theory")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()