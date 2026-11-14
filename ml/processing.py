from population import create_population
from evolution import simulate_evolution
from ml.dataset import create_dataset, prepare_ml_dataset
from analytics import plot_results, save_results


def run_experiments(n, positive_fitness=1.5, negative_fitness=0.5):
    results = []
    for exp in range(n):
        initial_population = create_population(1000)
        
        neutral_results = simulate_evolution(
            initial_population.copy(), "neutral", 100
        )

        positive_results = simulate_evolution(
                initial_population.copy(), "positive", 100, positive_fitness, negative_fitness
            )
        
        negative_results = simulate_evolution(
            initial_population.copy(), "negative", 100, positive_fitness, negative_fitness
        )
        
        results.append({
            "experiment": exp+1,
            "positive_fitness": positive_fitness,
            "negative_fitness": negative_fitness,
            "neutral_frequencies": neutral_results,
            "positive_frequencies": positive_results,
            "negative_frequencies": negative_results
        })

    return results


def generate_data_set():
    results = run_experiments(100)
    df = create_dataset(results)
    save_results(df, 'data/experiments_raw.csv')
    ml_df = prepare_ml_dataset(df)
    save_results(ml_df, 'data/experiments_ml.csv')

    print(ml_df.head())
    print(ml_df.shape)
    print(ml_df.isna().sum())
    print(ml_df['scenario'].value_counts())