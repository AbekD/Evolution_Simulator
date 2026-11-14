import pandas as pd

def create_dataset(results):
    data = []
    for experiment in results:
      for scenario in ["neutral", "positive", "negative"]:
          frequencies = experiment[f"{scenario}_frequencies"]

          for generation in range(len(frequencies)):
              data.append({
                  "experiment": experiment["experiment"],
                  "scenario": scenario,
                  "generation": generation + 1,
                  "frequency": frequencies[generation]
              })
    
    return pd.DataFrame(data)


def prepare_ml_dataset(df):
    data = []
    selected_generations = [1, 5, 10, 15, 20]

    for (experiment, scenario), group in df.groupby(["experiment", "scenario"]):
        row = {
            "experiment": experiment,
            "scenario": scenario
        }

        for generation in selected_generations:
            frequency = group.loc[group["generation"] == generation, "frequency"]
            row[f"freq_gen_{generation}"] = frequency.iloc[0]

        data.append(row)

    return pd.DataFrame(data)