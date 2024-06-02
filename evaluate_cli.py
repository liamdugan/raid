import argparse
import json

import pandas as pd

from raid.evaluate import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--results_path", type=str, required=True, help="Path to the detection result JSON to evaluate"
    )
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="Path to the dataset to evaluate for the results"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="results.json", help="Path to the output JSON to write scores"
    )
    parser.add_argument(
        "-t", "--target_fpr", type=float, default=0.05, help="Target false positive rate to evaluate detectors at"
    )
    args = parser.parse_args()

    print(f"Reading dataset at {args.data_path}...")
    df = pd.read_csv(args.data_path)

    print(f"Reading detection result at {args.results_path}...")
    with open(args.results_path) as f:
        d = json.load(f)

    print(f"Running evaluation...")
    evaluation_result = run_evaluation(d, df, args.target_fpr)

    print(f"Done! Writing evaluation result to output path: {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(evaluation_result, f)
