import argparse
import random
import re
from collections import Counter, defaultdict

import pandas as pd
from lingua import Language, LanguageDetectorBuilder
from model import is_api_model, is_chat_model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=True, help="The path of the dataframe to check")
parser.add_argument(
    "-t", "--templates", type=str, default="templates", help="The path to the folder of prompt templates"
)
parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    default="filtered.csv",
    help="The path to output file (used if apply_edits is active)",
)
parser.add_argument(
    "--apply_edits", action="store_true", help="Edit the dataset to fix all issues and output to outfile"
)

# Everything except "--check_null_generations" should pass before running generate.py
# After running generate.py, all checks should pass
parser.add_argument(
    "--check_null", action="store_true", help="Check that there are no null values in unexpected columns"
)
parser.add_argument("--check_null_generations", action="store_true", help="Check that there are no null generations")
parser.add_argument(
    "--check_no_duplicate_ids", action="store_true", help="Check there are no rows that have duplicate IDs"
)
parser.add_argument(
    "--check_no_duplicate_metadata", action="store_true", help="Check there are no rows that have duplicate metadata"
)
parser.add_argument("--check_correct_ids", action="store_true", help="Remove rows whose ids do not match our rules")
parser.add_argument("--check_defaults", action="store_true", help="Check the default metadata values are correct")
parser.add_argument("--check_prompts", action="store_true", help="Check that all prompts are correct")
parser.add_argument(
    "--check_adv_coverage", action="store_true", help="Remove sources that do not have a generation for all attacks"
)
parser.add_argument(
    "--check_coverage",
    action="store_true",
    help="Remove sources that do not have a generation for all models, decodings, and penalties",
)

# optional extra filtering checks
parser.add_argument("--check_domains_equal", action="store_true", help="Check that all domains are of equal size")
parser.add_argument(
    "--check_code_filtered", action="store_true", help="Check that code generations are filtered properly"
)
parser.add_argument(
    "--check_multilingual", action="store_true", help="Check that multilingual generations are of the correct language"
)
args = parser.parse_args()

tqdm.pandas()

df = pd.read_csv(args.data)

# Check that there are no null values in unexpected columns
if args.check_null:
    print("Checking for any null values in unexpected columns")

    def check_null(df, excluded_cols):
        cols_to_check = df.columns.difference(excluded_cols)
        return df[df[cols_to_check].isna().any(axis=1)]

    # Check for null values with separate excluded columns for human vs. non-human
    df_null_human = check_null(df[df["model"] == "human"], ["decoding", "repetition_penalty", "prompt"])
    df_null_non_human = check_null(df[df["model"] != "human"], ["generation"])
    df_null = pd.concat([df_null_human, df_null_non_human])

    # If any null columns were found, print
    if len(df_null) > 0:
        print(f"Found {len(df_null)} null values in unexpected columns")

        print("Metadata of null rows found...")
        print(df_null["model"].value_counts())
        print(df_null["decoding"].value_counts())
        print(df_null["repetition_penalty"].value_counts())
        print(df_null["domain"].value_counts())
        print(df_null["attack"].value_counts())

        if args.apply_edits:
            print("Deleting Null Rows")
            df = df[~df["id"].isin(df_null.id.unique().tolist())]

# Check that there are no null generations for non-human rows
if args.check_null_generations:
    print("Checking for any null generations")

    # Filter to only include non-human
    df_non_human = df[df["model"] != "human"]

    # If any null generations were found, print
    if len(df_null := df_non_human[df_non_human["generation"].isnull()]) > 0:
        print(f"Found {len(df_null)} null generations")

        print("Metadata of null generations found...")
        print(df_null["model"].value_counts())
        print(df_null["decoding"].value_counts())
        print(df_null["repetition_penalty"].value_counts())
        print(df_null["domain"].value_counts())
        print(df_null["attack"].value_counts())

        if args.apply_edits:
            print("Deleting null generations")
            df = df[~df["id"].isin(df_null.id.unique().tolist())]

# Check that there are no duplicate IDs
if args.check_no_duplicate_ids:
    print("Checking for generations with duplicate ids...")
    if len(df_duplicates := df[df.duplicated(subset=["id"])]):
        print(f"Found {len(df_duplicates)} rows with duplicate IDs...")

        print("Metadata of duplicate rows...")
        print(df_duplicates["model"].value_counts())
        print(df_duplicates["decoding"].value_counts())
        print(df_duplicates["repetition_penalty"].value_counts())
        print(df_duplicates["domain"].value_counts())
        print(df_duplicates["attack"].value_counts())

        if args.apply_edits:
            print("Dropping duplicate rows...")
            df = df.drop_duplicates(subset=["id"])

# Check that there are no generations with duplicate metadata
# (a row is considered a duplicate if all six metadata columns are identical)
if args.check_no_duplicate_metadata:
    metadata_cols = ["source_id", "domain", "model", "decoding", "repetition_penalty", "attack"]
    print("Checking for generations with duplicate metadata...")
    if len(df_duplicates := df[df.duplicated(subset=metadata_cols)]):
        print(f"Found {len(df_duplicates)} rows with duplicate metadata...")

        print("Metadata of duplicate rows...")
        print(df_duplicates["model"].value_counts())
        print(df_duplicates["decoding"].value_counts())
        print(df_duplicates["repetition_penalty"].value_counts())
        print(df_duplicates["domain"].value_counts())
        print(df_duplicates["attack"].value_counts())

        if args.apply_edits:
            print("Dropping duplicate rows...")
            df = df.drop_duplicates(subset=metadata_cols)

# Check if IDs have the correct scheme
#  - If attack is none then id = adv_source_id
#  - If attack is none and model is human, id = adv_source_id = source_id
if args.check_correct_ids:
    print("Checking IDs have the correct scheme...")
    df_noattack = df[df["attack"] == "none"]
    ids_to_remove = []
    for i, row in tqdm(df_noattack.iterrows(), total=len(df_noattack)):
        if row["id"] != row["adv_source_id"]:
            print(f"Error: ID {row['id']} doesnt match adv source id {row['adv_source_id']} for attack none")
            ids_to_remove.append(row["id"])
        if row["model"] == "human" and row["id"] != row["source_id"]:
            print(f"Error: Human Gen ID {row['id']} not same as source id {row['source_id']} for attack none")
            ids_to_remove.append(row["id"])

    if args.apply_edits:
        print(f"Dropping {len(ids_to_remove)} rows with incorrect ID scheme...")
        df = df[~df["id"].isin(ids_to_remove)]

# Check that default values are correct
# - If human generation: decoding, repetition_penalty, prompt = NaN
if args.check_defaults:
    print("Checking rows have correct default values...")
    ids_to_remove = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row["model"] == "human":
            if not (pd.isna(row["decoding"]) and pd.isna(row["repetition_penalty"]) and pd.isna(row["prompt"])):
                ids_to_remove.append(row["id"])

    if len(ids_to_remove) > 0:
        print(f"Found {len(ids_to_remove)} rows with incorrect default values...")
        df_bad_defaults = df[df["id"].isin(ids_to_remove)]

        print("Metadata of rows with incorrect default values...")
        print(df_bad_defaults["model"].value_counts())
        print(df_bad_defaults["decoding"].value_counts())
        print(df_bad_defaults["repetition_penalty"].value_counts())
        print(df_bad_defaults["domain"].value_counts())
        print(df_bad_defaults["attack"].value_counts())

        if args.apply_edits:
            print(f"Dropping {len(ids_to_remove)} rows with incorrect default values...")
            df = df[~df["id"].isin(ids_to_remove)]

# Check that prompts are correct
if args.check_prompts:
    print("Checking that all prompts are correct...")

    # Filter to no human (since all human prompts are NaN)
    df_no_human = df[df["model"] != "human"]

    # Create a dictionary of prompt templates for each domain in the dataset
    templates_dict = defaultdict(dict)
    for d in df_no_human["domain"].unique().tolist():
        for template_type in ["chat", "completion"]:
            with open(f"{args.templates}/{template_type}/{d}.txt") as f:
                templates_dict[d][template_type] = "".join(f.readlines())

    # Check that all rows have correct prompts
    ids_to_remove = []
    for i, row in tqdm(df_no_human.iterrows(), total=len(df_no_human)):
        template_type = "chat" if is_chat_model(row["model"]) else "completion"
        template = templates_dict[row["domain"]][template_type]
        prompt = template.format(title=row["title"])
        if prompt != row["prompt"]:
            ids_to_remove.append(row["id"])

    # If we find rows with incorrect prompts, print them and remove them if apply_edits = true
    if len(ids_to_remove) > 0:
        print(f"Found {len(ids_to_remove)} rows with incorrect prompts...")
        df_bad_prompts = df[df["id"].isin(ids_to_remove)]

        print("Metadata of rows with incorrect prompts...")
        print(df_bad_prompts["model"].value_counts())
        print(df_bad_prompts["decoding"].value_counts())
        print(df_bad_prompts["repetition_penalty"].value_counts())
        print(df_bad_prompts["domain"].value_counts())
        print(df_bad_prompts["attack"].value_counts())

        if args.apply_edits:
            print(f"Dropping {len(ids_to_remove)} rows with incorrect prompts...")
            df = df[~df["id"].isin(ids_to_remove)]

# For each adv_source_id there is exactly one row per attack
if args.check_adv_coverage:
    print("Checking that all texts have exactly one gen for each adversarial attack...")
    asid_counter = Counter(df["adv_source_id"].tolist())
    num_attacks = len(df["attack"].unique().tolist())

    ids_with_issues = [x for x, y in asid_counter.items() if y != num_attacks]
    if len(ids_with_issues) > 0:
        print(f"Found {len(ids_with_issues)} ids with less than {num_attacks} adversarial rows")
        print("Metadata...")
        print(df[df["adv_source_id"].isin(ids_with_issues)].attack.value_counts())

    # Remove all affected IDs
    if args.apply_edits:
        print(f"Dropping {len(ids_with_issues)} rows that are missing all adversarial attacks...")
        df = df[~df["adv_source_id"].isin(ids_with_issues)]

# For each source_id check there is exactly one generation for each model, decoding, and repetition_penalty
#
# NOTE: This doesn't actually check each decoding and repetition penalty individually to save time.
#       This means that it can be wrong if there exists any generations with duplicate dec/rep,
#       for this reason make sure that check_no_duplicates comes up clean before running this.
if args.check_coverage:
    print("Checking that all sources have exactly one gen for each model, decoding, and repetition_penalty...")
    models = df[df["model"] != "human"]["model"].unique().tolist()
    api_models = list(filter(is_api_model, models))

    # Calculate exactly how many generations we should expect for each human source
    # (We expect 2 for each api model and 4 for each HF model due to repetition penalties)
    # (Final +1 is for the source generation itself)
    num_gens_expected = sum([2 if m in api_models else 4 for m in models]) + 1
    sid_counter = Counter(df[df["attack"] == "none"]["source_id"].tolist())

    ids_with_issues = [x for x, y in sid_counter.items() if y != num_gens_expected]
    if len(ids_with_issues) > 0:
        print(f"Found {len(ids_with_issues)} ids with less than {num_gens_expected} rows")
        print("Metadata...")
        print(df[df["source_id"].isin(ids_with_issues)].model.value_counts())
        print(df[df["source_id"].isin(ids_with_issues)].decoding.value_counts())
        print(df[df["source_id"].isin(ids_with_issues)].repetition_penalty.value_counts())

    # Remove all affected IDs
    if args.apply_edits:
        print(f"Removing {len(ids_with_issues)} Rows")
        df = df[~df["source_id"].isin(ids_with_issues)]

# Check that all domains are of equal size
if args.check_domains_equal:
    print("Checking that all domains have an equal amount of source IDs...")
    df_human = df[df["model"] == "human"]
    counts = df_human["domain"].value_counts().to_dict()

    # Print domains that aren't equal (if they exist)
    value_counts = [v for k, v in counts.items()]
    if not all(v == value_counts[0] for v in value_counts):
        print("Found domains with unequal amounts...")
        print(counts)

    if args.apply_edits:
        # Set random seed for reproducibility
        random.seed(42)

        # Find the domain with the lowest count in the dataset
        min_key = min(counts, key=counts.get)

        # For each domain randomly select some amount of source_ids to delete
        for k, v in counts.items():
            num_ids_to_remove = v - counts[min_key]
            all_source_ids = df[df["domain"] == k]["source_id"].unique().tolist()

            # Randomly select the source_ids to delete
            source_ids_to_remove = random.sample(all_source_ids, num_ids_to_remove)
            df = df[~df["source_id"].isin(source_ids_to_remove)]

# Check that code is properly filtered
if args.check_code_filtered:
    print("Checking that all code generations are properly filtered...")

    def filter_code(row):
        if row["domain"] == "code":
            if result := re.search("```(python)?(.*?)```", row["generation"], flags=re.DOTALL):
                g = list(result.groups())[-1].strip()
                print(f"Found unfiltered code generation at ID:{row['id']}...")
                return [g, len(g)] if args.apply_edits else [row["generation"], row["length"]]
        return [row["generation"], row["length"]]

    df[["generation", "length"]] = df.progress_apply(filter_code, axis=1, result_type="expand")

# Check that multilingual data is actually multilingual
if args.check_multilingual:
    print("Checking to make sure multilingual data is the correct language...")
    german_detector = LanguageDetectorBuilder.from_languages(*[Language.ENGLISH, Language.GERMAN]).build()
    czech_detector = LanguageDetectorBuilder.from_languages(*[Language.ENGLISH, Language.CZECH]).build()

    detector_dict = {
        "german": (german_detector, Language.GERMAN),
        "czech": (czech_detector, Language.CZECH),
    }

    def match_language(row):
        if row["domain"] not in ["german", "czech"]:
            return True

        # Get the detector and language for the domain
        detector, language = detector_dict[row["domain"]]

        # Apply the language detector
        det_value = detector.compute_language_confidence(row["generation"], language)

        # If the detector output doesn't match the correct language, print
        if det_value < 1.0:
            print(f"Found non-{row['domain']} generation at ID:{row['id']}, {det_value}...")

        return (det_value == 1.0) if args.apply_edits else True

    df = df[df.progress_apply(match_language, axis=1)]

if args.apply_edits:
    print("Final Metadata of Dataframe")
    print(df["model"].value_counts())
    print(df["decoding"].value_counts())
    print(df["domain"].value_counts())
    print(df["repetition_penalty"].value_counts())
    print(df["attack"].value_counts())

    # Output to csv file
    print(f"Writing new csv to {args.output_path}...")
    df.to_csv(args.output_path, index=False)
