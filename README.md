<<<<<<< HEAD
# Evolution Simulator: The Evolution of an Unusual Trait

An educational Python project that simulates evolutionary processes and applies Machine Learning to analyze genetic changes across generations.

The project combines population genetics, evolutionary simulation, statistical analysis, data visualization, and Machine Learning.

## 1. Project Overview

The Evolution Simulator explores how inherited traits change in a population over multiple generations under different evolutionary conditions.

The project uses a simplified artificial genetic model of an unusual inherited trait: an enlarged hand.

The main goals are:
- Simulate genetic inheritance and reproduction.
- Investigate the effects of natural selection.
- Explore genetic drift in populations of different sizes.
- Analyze changes in allele and phenotype frequencies.
- Apply Machine Learning to classify evolutionary scenarios.
- Compare simulation results with theoretical expectations.

**Note:** The genetic model is intentionally simplified for educational purposes. It does not represent the actual inheritance of human hand size.

## 2. Genetic Model

The simulation uses two alleles:

| Allele | Description |
|---|---|
| A | Normal hand size |
| a | Enlarged hand |

The enlarged-hand trait follows a simplified recessive inheritance model.

| Genotype | Phenotype |
|---|---|
| AA | Normal hand |
| Aa | Normal hand |
| aa | Enlarged hand |

An individual expresses the enlarged-hand trait only when both alleles are `a`.

The initial allele probabilities can be configured. By default:

- P(A) = 0.5
- P(a) = 0.5

## 3. Evolutionary Mechanisms

### 3.1 Natural Selection

The original simulation includes three scenarios:

| Scenario | Description | Fitness of aa |
|---|---|---:|
| Neutral | No selection advantage or disadvantage | 1.0 |
| Positive | The trait has higher reproductive success | 1.5 |
| Negative | The trait has lower reproductive success | 0.5 |

Individuals are selected as parents according to their relative fitness.

Fitness values represent hypothetical reproductive-success weights, not direct probabilities.

### 3.2 Genetic Drift

Genetic Drift is a separate simulation that investigates random changes in allele frequencies between generations.

Unlike natural selection, genetic drift does not favor any particular allele. Every individual has equal reproductive fitness.

The simulation uses different population sizes:

- 50 individuals
- 100 individuals
- 500 individuals
- 1000 individuals

Each experiment tracks:
- Allele frequency of `a`.
- Frequency of the recessive phenotype `aa`.
- Changes across generations.
- Allele fixation and loss.

The simulation records generation 0 as the initial population and then tracks changes over 100 generations.

## 4. Wright–Fisher Model

The genetic drift simulation follows the Wright–Fisher model.

Each new generation is formed by randomly sampling parental alleles. The population size remains constant.

The theoretical variance of allele frequency under neutral drift is:

\[
Var(q_t)=q_0(1-q_0)
\left(1-\left(1-\frac{1}{2N}\right)^t\right)
\]

Here:
- \(q_0\) is the initial allele frequency.
- \(N\) is the population size.
- \(t\) is the number of generations.

The simulation results are compared with theoretical expectations to evaluate the behavior of the model.

## 5. Machine Learning

The project includes a Machine Learning component that classifies evolutionary scenarios based on simulated trait frequencies.

### Dataset Generation

Simulation results are transformed into a structured dataset.

The ML dataset uses five features:

- `freq_gen_1`
- `freq_gen_5`
- `freq_gen_10`
- `freq_gen_15`
- `freq_gen_20`

The target variable is the evolutionary scenario:

- `neutral`
- `positive`
- `negative`

### Models

Two classification algorithms are implemented:

1. Logistic Regression
2. Random Forest Classifier

### Evaluation

The ML pipeline uses:

- 80/20 train-test split.
- GroupShuffleSplit to prevent data leakage between experiments.
- GroupKFold cross-validation.
- Accuracy.
- Precision, Recall and F1-score.
- Confusion Matrix.
- Feature Importance for Random Forest.

In the initial synthetic dataset, the Random Forest achieved 100% accuracy on the held-out test set and 100% accuracy across the five cross-validation folds.

These results describe performance on the generated simulation data and do not establish equivalent performance on real biological data.

## 6. Genetic Drift Experiment Results

The following results were obtained using 30 experiments per population size, 100 generations, and random seed 42.

| Population Size | Mean `aa` Frequency |
|---:|---:|
| 50 | 42.9% |
| 100 | 38.5% |
| 500 | 33.9% |
| 1000 | 26.5% |

The results demonstrate that smaller populations experience stronger random fluctuations in allele frequencies.

An interesting observation is that the average frequency of the recessive phenotype can increase even without positive natural selection.

This happens because the phenotype frequency depends on the square of the recessive allele frequency:

\[
E[q^2]=E[q]^2+Var(q)
\]

As genetic drift increases the variance of allele frequencies, the average frequency of the recessive phenotype can also increase.

The experimental values are statistical estimates based on 30 independent simulations per population size.

## 7. Visualizations

The project generates several plots:

1. **Genetic Drift Trajectories** — shows individual simulation runs and their average trajectory for different population sizes.
2. **Mean Trait Frequency Comparison** — compares average phenotype frequencies and variation between population sizes.
3. **Variance vs. Theory** — compares simulated allele-frequency variance with the theoretical Wright–Fisher expectation.

These visualizations help investigate the relationship between population size and genetic drift.

## 8. Project Structure

```text
evolution-simulator/
│
├── main.py
├── evolution.py
├── genetics.py
├── population.py
├── analytics.py
├── drift.py
├── test_drift.py
│
├── data/
│   ├── evolution_results.csv
│   ├── experiments_raw.csv
│   ├── experiments_ml.csv
│   └── genetic_drift_results.csv
│
└── ml/
    ├── dataset.py
    ├── model.py
    └── processing.py
```

### Main Modules

| File | Responsibility |
|---|---|
| `main.py` | Runs the existing ML workflow |
| `genetics.py` | Genetic inheritance and phenotype determination |
| `population.py` | Population generation |
| `evolution.py` | Natural selection and reproduction |
| `analytics.py` | Statistical analysis, plots and CSV export |
| `drift.py` | Genetic drift experiments and theoretical comparison |
| `test_drift.py` | Basic tests for the genetic drift extension |
| `ml/dataset.py` | Dataset generation and preparation |
| `ml/model.py` | ML model training and evaluation |
| `ml/processing.py` | Simulation experiment and dataset preparation workflow |

## 9. Installation

Clone the repository:

```bash
git clone <repository-url>
cd evolution-simulator
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 10. Running the Project

### Run the original ML pipeline

```bash
python main.py
```

This loads the existing ML dataset, trains the Random Forest classifier, evaluates it, and performs additional experiments using different fitness parameters.

### Run Genetic Drift tests

```bash
python test_drift.py
```

### Run a quick Genetic Drift experiment

```bash
python drift.py 3
```

This runs 3 experiments per population size.

### Run the full Genetic Drift experiment

```bash
python drift.py 30 42
```

This runs 30 experiments per population size using random seed 42.

The experiment saves its results to:

```text
data/genetic_drift_results.csv
```

The plots are displayed sequentially. Close each plot window to continue to the next one.

Run commands from the project root directory.

## 11. Technologies

- **Python** — main programming language.
- **NumPy** — random sampling and numerical calculations.
- **Pandas** — data processing and CSV datasets.
- **Matplotlib** — data visualization.
- **Scikit-learn** — Machine Learning models and evaluation.

## 12. Current Project Status

Implemented features:

- [x] Genetic inheritance model.
- [x] Population generation.
- [x] Natural selection simulation.
- [x] Neutral, positive and negative selection scenarios.
- [x] Configurable initial allele probabilities.
- [x] Genetic Drift simulation.
- [x] Population-size experiments.
- [x] Allele and phenotype frequency tracking.
- [x] Allele fixation and loss analysis.
- [x] Wright–Fisher theoretical comparison.
- [x] CSV export.
- [x] Data visualization.
- [x] ML dataset preparation.
- [x] Logistic Regression.
- [x] Random Forest classification.
- [x] Model evaluation and cross-validation.
- [x] Basic Genetic Drift tests.

## 13. Future Improvements

Potential directions for further development:

- Integrate real population genetics datasets.
- Implement gene flow and mutation.
- Investigate additional population genetics models.
- Extend Machine Learning experiments to predict population size from genetic drift patterns.
- Improve simulation performance through vectorization.
- Build an interactive visualization interface.

---

**Project Purpose:** An educational exploration of evolutionary mechanisms through computational simulation, statistical analysis, and Machine Learning.
