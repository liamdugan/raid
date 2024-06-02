import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Path to input CSV file")
parser.add_argument("-n", "--num_shards", type=int, required=True, help="Number of shards")
parser.add_argument("-d", "--shards_dir", type=str, required=True, help="Folder to save the shards to")
parser.add_argument(
    "--split_on_sources", action="store_true", help="Split data by selecting which source documents go in each shard"
)
args = parser.parse_args()

df = pd.read_csv(args.input)

if split_on_sources:
    source_ids = df["source_id"].unique().tolist()
    n = int(len(source_ids) / args.num_shards)
    sources = [source_ids[i : i + n] for i in range(0, len(source_ids), n)]
    shards = [df[df["source_id"].isin(x)] for x in sources]
else:
    n = int(df.shape[0] / args.num_shards)
    shards = [df[i : i + n] for i in range(0, df.shape[0], n)]

name = os.path.basename(args.input)

for i, shard in enumerate(shards):
    outpath = os.path.join(args.shards_dir, f"{i}_{name}")
    print(f"Writing shard {i} to file {outpath}..")
    shard.to_csv(outpath, index=False)
