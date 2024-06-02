import argparse
import json

import pandas as pd

from detectors.detector import get_detector
from raid.detect import run_detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the detector model you wish to run")
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="Specify path to the data file to run detection on"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="predictions.json", help="The file name to write the results to"
    )
    args = parser.parse_args()

    print(f"Loading dataset with name {args.data_path}...")
    df = pd.read_csv(args.data_path)

    print(f"Loading detector with name {args.model}...")
    detector = get_detector(args.model)

    print(f"Running detection...")
    results = run_detection(detector.inference, df)

    print(f"Done! Writing predictions to output path: {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(results, f)
