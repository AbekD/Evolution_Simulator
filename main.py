import pandas as pd
import sys
import os

from ml.model import split_dataset, train_random_forest, evaluate_model, cross_validate_model
from ml.processing import run_experiments
from ml.dataset import create_dataset, prepare_ml_dataset
from analytics import (
    plot_results,
    save_results,
    plot_drift_trajectories,
    plot_drift_mean_comparison,
    plot_variance_comparison
)
from drift import run_drift_experiments, summarize_drift, variance_comparison


def main():
    ml_df = pd.read_csv("data/experiments_ml.csv")
    df = pd.read_csv("data/experiments_raw.csv")

    plot_results(df)

    X_train, X_test, y_train, y_test = split_dataset(ml_df)

    rf_model = train_random_forest(X_train, y_train)
    print("model trained")

    evaluate_model(rf_model, X_test, y_test)

    test_results = run_experiments(
        5,
        positive_fitness=2.0,
        negative_fitness=0.2
    )
    
    new_df = create_dataset(test_results)
    new_ml_df = prepare_ml_dataset(new_df)

    X_test_1 = new_ml_df.drop(columns=["scenario", "experiment"])
    y_test_1 = new_ml_df["scenario"]

    evaluate_model(rf_model, X_test_1, y_test_1)

    test_results_2 = run_experiments(
        10,
        positive_fitness=1.7,
        negative_fitness=0.4
    )

    new_df1 = create_dataset(test_results_2)
    new_ml_df1 = prepare_ml_dataset(new_df1)
    
    X_test_2 = new_ml_df1.drop(columns=["scenario", "experiment"])
    y_test_2 = new_ml_df1["scenario"]
    
    evaluate_model(rf_model, X_test_2, y_test_2)
    
def run_genetic_drift(experiments=30, seed=42):
    path = "data/genetic_drift_results.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = run_drift_experiments(experiments=experiments, seed=seed)
        save_results(df, path)

    print(summarize_drift(df).round(4).to_string(index=False))

    plot_drift_trajectories(df)
    plot_drift_mean_comparison(df)
    plot_variance_comparison(variance_comparison(df))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drift":
        run_genetic_drift()
    else:
        main()