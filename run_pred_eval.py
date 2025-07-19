import argparse
import json

import pandas as pd

from detectors.detector import get_detector
from raid.detect import run_detection
from raid.evaluate import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the detector model you wish to run")
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="Specify path to the data file to run detection on"
    )
    parser.add_argument(
        "-p", "--output_preds_path", type=str, default="predictions.json", help="The file name to write the results to"
    )
    parser.add_argument(
        "-e", "--output_eval_path", type=str, default="results.json", help="The file name to write the results to"
    )
    args = parser.parse_args()

    print("Loading MAGE dataset...")
    df = pd.read_csv(args.data_path)

    # Add dummy columns for attack, decode, repetition_penalty, include
    for k, v in [('attack', 'none'), ('decoding', None), ('repetition_penalty', None)]:
        if k not in df.columns:
            print(f"Failed to find {k}, adding value {repr(v)}")
            df[k] = v

    print(f"Loading detector with name {args.model}...")
    detector = get_detector(args.model)

    print(f"Running detection...")
    results = run_detection(detector.inference, df)

    print(f"Done! Writing predictions to output path: {args.output_preds_path}")
    with open(args.output_preds_path, "w") as f:
        json.dump(results, f)

    # if "label" not in df.columns:
    #     print("No labels found in dataframe; evaluation will not be run.")
    #     exit(0)

    print("Running evaluation on the predictions...")
    evaluation_result = run_evaluation(results, df)

    print(f"Writing evaluation results to {args.output_eval_path}...")
    with open(args.output_eval_path, "w") as f:
        json.dump(evaluation_result, f, indent=4)