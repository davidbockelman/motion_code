"""
Batch runner for test_mixture_experts.py across multiple datasets, m, and R values.

For each dataset, runs all combinations of m and R, repeating each combination
3 times, and appends results to per-dataset JSON files.
"""
import os
import subprocess

# Datasets to evaluate (basic set)
DATASETS = [
    "Chinatown",
    "ECGFiveDays",
    "FreezerSmallTrain",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "ItalyPowerDemand",
    "Lightning7",
    "MoteStrain",
    "PowerCons",
    "SonyAIBORobotSurface2",
    "UWaveGestureLibraryAll",
    "GunPointOldVersusYoung",
]

# Hyperparameter combinations
M_VALUES = [5, 10]
R_VALUES = [2, 3]

# Number of repetitions per combination
REPEATS = 3


def run_command(cmd):
    """Run a shell command and stream output."""
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    os.makedirs("results", exist_ok=True)

    for dataset in DATASETS:
        json_path = os.path.join("results", f"{dataset}.json")
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset}")
        print(f"Results file: {json_path}")
        print("=" * 80)

        for m in M_VALUES:
            for R in R_VALUES:
                for rep in range(1, REPEATS + 1):
                    print(f"\n--- Run {rep}/{REPEATS} | dataset={dataset}, m={m}, R={R} ---")
                    cmd = [
                        "python",
                        "test_mixture_experts.py",
                        "--dataset",
                        dataset,
                        "--m",
                        str(m),
                        "--R",
                        str(R),
                        "--json-file",
                        json_path,
                    ]
                    run_command(cmd)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()

