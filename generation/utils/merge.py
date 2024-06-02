import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--files", type=str, nargs="+", required=True, help="The paths to the input CSV files to merge"
)
parser.add_argument(
    "-o", "--output_path", type=str, default="out.csv", help="The name of the output csv file to write to"
)
args = parser.parse_args()

# Read in all of the csv files
print("Reading dataframes...")
dfs = [pd.read_csv(f) for f in args.files]

# Concatenate into one big dataframe
print("Concatenating dataframes...")
df = pd.concat(dfs)

# Output to csv file
df.to_csv(args.output_path, index=False)
