import argparse
import random

import numpy as np
import pandas as pd
from model import get_model, get_models_list
from tqdm import tqdm


def regenerate(df, model, decoding, penalty, batch_size=20, max_retries=5):
    # Create the bitmask for the model, decoding strategy, and repetition penalty
    bitmask = (df["model"] == model) & (df["decoding"] == decoding) & (df["repetition_penalty"] == penalty)

    # If initial null rows is 0 then return the dataframe
    if (initial_null_rows := len(df[bitmask & df["generation"].isnull()])) == 0:
        return df

    # Initialize the generator and get the decoding strategy
    generator = get_model(model)
    config = generator.get_config(decoding, penalty)

    # Initialize our retry count for this model
    retries = 0

    with tqdm(total=initial_null_rows) as pbar:
        # while there are still null rows for the model and decoding strategy
        while null_rows := len(df[bitmask & df["generation"].isnull()]):
            # Randomly sample some number of the remaining null rows
            indices = df.index[bitmask & df["generation"].isnull()].tolist()
            sample = random.sample(indices, min(batch_size, len(indices)))

            # Generate outputs for the rows given the prompts
            generations = generator.generate(df.loc[sample]["prompt"].tolist(), config)

            # If generations are of zero length, convert them to nan
            generations = [g if len(g) > 0 else np.nan for g in generations]

            # Add the new rows to the dataframe
            df.loc[sample, "generation"] = generations

            # Update the progress bar with the new rows added
            pbar.update(rows_added := null_rows - len(df[bitmask & df["generation"].isnull()]))

            # If no new rows added, increment the retry counter
            retries = (retries + 1) if not rows_added else 0

            # If we are over the maximum retries allowed, break
            if retries >= max_retries:
                break

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="The path to the input CSV file")
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=get_models_list() + ["all"],
        help="The models to generate for.",
    )
    parser.add_argument(
        "-c", "--cache_outputs", action="store_true", help="Periodically cache outputs after a successful model run"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="The batch size for generation")
    parser.add_argument(
        "-r", "--max_retries", type=int, default=5, help="The maximum number of retries for a failed generation"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="generated_data.csv", help="The name of the file to write to"
    )
    args = parser.parse_args()

    # Read in the existing dataframe of all generations
    df = pd.read_csv(args.input)

    # Determine the models to regenerate for
    models = get_models_list() if "all" in args.models else args.models

    # Regenerate the data for all models selected for all decoding strategies w/ all repetition penalties
    for m in models:
        for d in ["greedy", "sampling"]:
            for r in ["yes", "no"]:
                print(f"Generating for model {m} with decoding strategy {d} and penalty {r}")
                df = regenerate(df, m, d, r, args.batch_size, args.max_retries)

                # If we're caching outputs, write the intermediate outputs to the csv
                if args.cache_outputs:
                    print("Writing intermediate outputs to cache")
                    df.to_csv(args.output_path, index=False, escapechar="\\")

    # Write the final output to a csv
    df.to_csv(args.output_path, index=False, escapechar="\\")
