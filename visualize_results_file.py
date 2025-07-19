import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--results_file",
    type=str,
    required=True,
    help="Specify the results file to visualize",
)
args = parser.parse_args()

with open(args.results_file) as f:
    d = json.load(f)

df = pd.DataFrame(d['scores'])

for d in df.domain.unique():
    accuracy = df[(df['model'] == 'all') & (df['domain'] == d) & (df['attack'] == 'none') & (df['decoding'] == 'all') & (df['repetition_penalty'] == 'all')]['auroc']
    try:
        print(f"{d} -- {accuracy.item()}")
    except Exception:
        pass

# for m in df.model.unique():
#     accuracy = df[(df['model'] == m) & (df['domain'] == 'all') & (df['attack'] == 'none') & (df['decoding'] == 'all') & (df['repetition_penalty'] == 'all')]['auroc']
#     try:
#         print(f"{m} -- {accuracy.item()}")
#     except Exception:
#         pass
#print(df)

# # Conditions
# all_model_results = [result for result in d['scores'] if result['model'] == 'all']


# print([result['accuracy'] for result in d['scores'] if result['domain'] == 'wp' and result['model'] == 'all'])