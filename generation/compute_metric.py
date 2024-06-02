import argparse
import json
import time
from datetime import date

import pandas as pd
from metrics.metric import get_metric
from tqdm import tqdm


def get_date_time() -> str:
    today = date.today()
    current_date = today.strftime("%b-%d-%Y")

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    current_time = current_time.replace(":", "-")

    date_time = current_date + "-" + current_time
    return date_time


tqdm.pandas()


def run_compute_metric(metric_name, data_path):
    # Load the metric
    metric = get_metric(metric_name)

    # Load the dataset
    df = pd.read_csv(data_path)

    # Run metric on all items in the dataset and put output in column score
    if metric_name == "tokens":
        df["score"] = metric.compute(df["generation"], df["model"])
    else:
        df["score"] = metric.compute(df["generation"])

    # Convert scores and ids to dict in 'records' format for seralization
    # e.g. [{'id':'...', 'score':0}, {'id':'...', 'score':1}, ...]
    results = df[["id", "score"]].to_dict(orient="records")

    # Create the Metrics JSON
    evaluation_result = {"date": get_date_time(), "dataset_path": data_path, "metric": metric_name, "results": results}

    return evaluation_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_name", type=str, required=True, help="specify the metric you wish to run")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Specify path to the data file to compute metrics on"
    )
    parser.add_argument(
        "--output_fname", type=str, required=True, default="results.json", help="The filename to write the results to"
    )
    args = parser.parse_args()

    print(f"Running Metric {args.metric_name} on data {args.data_path}...")
    evaluation_result = run_compute_metric(args.metric_name, args.data_path)

    print(f"Done! Writing predictions to output path: {args.output_fname}")
    with open(args.output_fname, "w") as f:
        json.dump(evaluation_result, f)
