import argparse
import random
import uuid

import pandas as pd
from attack import get_attack
from tqdm import tqdm


def process(df, infile, attacker_name):
    # Get the adversarial attack
    attacker = get_attack(attacker_name)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Apply the adversarial attack
        outputs = attacker.attack(row["generation"])

        # Set metadata to new values for the generation
        outputs["attack"] = attacker_name
        outputs["id"] = uuid.uuid4()

        # Update the dataframe with the outputs of the attack
        for k, v in outputs.items():
            df.loc[i, k] = str(v)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("-a", "--attacker", type=str, required=True, help="Name of the attacker to use")
    parser.add_argument(
        "-o", "--output_path", type=str, default="adversarial.csv", help="The name of the file to write to"
    )
    args = parser.parse_args()

    # Read in the existing dataframe of all generations
    df = pd.read_csv(args.input)

    # Set random seed for reproducibility
    random.seed(42)

    # Apply the adversarial attack to the dataframe
    adversarial_df = process(df, args.input, args.attacker)

    # Write the output to a csv
    adversarial_df.to_csv(args.output_path, index=False)
