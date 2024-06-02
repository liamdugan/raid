import argparse

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", type=str, required=True, help="The path to the dataframe file into which you want to replace rows"
)
parser.add_argument(
    "-i", "--input", type=str, required=True, help="The path to the dataframe containing the new versions of the rows"
)
parser.add_argument(
    "-o", "--output_path", type=str, default="out.csv", help="The name of the output csv file to write to"
)
args = parser.parse_args()

tqdm.pandas()

df = pd.read_csv(args.data)
df_new = pd.read_csv(args.input)

# Make a set of ids from the new dataframe
ids_in_df_new = set(df_new["id"].unique().tolist())


# Go through the dataframe and replace rows if the IDs match
def replace_rows(row):
    if row["id"] in ids_in_df_new:
        return df_new[df_new["id"] == row["id"]].iloc[0]
    else:
        return row


print("Updating old df with new rows")
df = df.progress_apply(replace_rows, axis=1, result_type="expand")

print("Writing to output csv file...")
df.to_csv(args.output_path, index=False)
