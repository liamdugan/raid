import argparse
import uuid
from collections import defaultdict

import pandas as pd

# List of domains in our human-written data
ALL_DOMAINS = [
    "abstracts",
    "books",
    "code",
    "german",
    "czech",
    "news",
    "poetry",
    "recipes",
    "reddit",
    "reviews",
    "wiki",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--num_samples", type=int, default=1000, help="The number of human-written examples to include"
)
parser.add_argument(
    "-p",
    "--sources_path",
    type=str,
    default="sources",
    help="The path to the folder that contains the human data (default ./sources)",
)
parser.add_argument(
    "-d",
    "--domains",
    type=str,
    nargs="+",
    required=True,
    choices=ALL_DOMAINS + ["all"],
    help="The domains to sample for the dataframe",
)
parser.add_argument(
    "-o", "--output_path", type=str, default="out.csv", help="The path to write the sampled dataframe to"
)
args = parser.parse_args()

# Determine the domains to sample for
domains = ALL_DOMAINS if "all" in args.domains else args.domains

# Initialize the dictionary containing the rows
rows = defaultdict(list)

for domain in domains:
    # Read in the human-written data files up to num_samples
    df = pd.read_csv(f"{args.sources_path}/{domain}/2000_{domain}.csv")[: args.num_samples]

    # For each prompt, create the row and add it to the new dataframe dict
    for index, row in df.iterrows():
        source_id = uuid.uuid4()
        row = {
            "id": source_id,
            "adv_source_id": source_id,
            "source_id": source_id,
            "model": "human",
            "decoding": "",
            "repetition_penalty": "",
            "attack": "none",
            "domain": domain,
            "title": row["title"],
            "prompt": "",
            "generation": row["text"],
            "num_edits": 0,
            "edits": "[]",
        }
        for k, v in row.items():
            rows[k].append(v)

# Create the human data dataframe from the dict and write to the output path
df = pd.DataFrame.from_dict(rows)
df.to_csv(args.output_path, index=False)
