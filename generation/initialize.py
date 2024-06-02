import argparse
import uuid
from collections import defaultdict

import pandas as pd
from model import get_models_list, is_api_model, is_chat_model

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--df_path", type=str, required=True, help="The path of the dataframe to add the model to")
parser.add_argument(
    "-m",
    "--models",
    type=str,
    nargs="+",
    required=True,
    choices=get_models_list() + ["all"],
    help="The models to generate for.",
)
parser.add_argument("-t", "--templates", type=str, default="templates", help="The path to the folder of templates")
parser.add_argument("-o", "--output_path", type=str, default="init.csv", help="The name of the file to write to")
args = parser.parse_args()


def get_template(model: str, domain: str):
    template_type = "chat" if is_chat_model(model) else "completion"
    with open(f"{args.templates}/{template_type}/{domain}.txt") as f:
        return "".join(f.readlines())


def apply_template(title: str, template: str):
    return template.format(title=title)


# Read in the existing dataframe and grab only the human rows
df = pd.read_csv(args.df_path)
df_human = df[df["model"] == "human"]

# Determine the models to regenerate for
models = get_models_list() + ["all"] if "all" in args.models else args.models

# Only add models that do not already exist in the dataframe
models = [m for m in models if m not in set(df["model"])]

# For each model, for each human-written text, for each sampling strategy
# create the prompt and add the row to the dataframe
model_rows = defaultdict(list)
for m in models:
    penalties = ["no"] if is_api_model(m) else ["yes", "no"]
    for index, row in df_human.iterrows():
        template = get_template(m, row["domain"])
        prompt = apply_template(row["title"], template)
        for s in ["greedy", "sampling"]:
            for p in penalties:
                gen_id = uuid.uuid4()
                model_row = {
                    "id": gen_id,
                    "adv_source_id": gen_id,
                    "source_id": row["id"],
                    "model": m,
                    "decoding": s,
                    "repetition_penalty": p,
                    "attack": "none",
                    "domain": row["domain"],
                    "title": row["title"],
                    "prompt": prompt,
                    "generation": "",
                    "num_edits": 0,
                    "edits": "[]",
                }
                for k, v in model_row.items():
                    model_rows[k].append(v)

# Convert the dictionary of new rows into a dataframe and concat it to the existing df
model_df = pd.DataFrame.from_dict(model_rows)
new_df = pd.concat([df, model_df])

# Write the df to a csv file
new_df.to_csv(args.output_path, index=False)
