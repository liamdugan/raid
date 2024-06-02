import argparse
import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", type=str, nargs="+", required=True, help="The paths/patterns to the input JSONs")
parser.add_argument("-g", "--use_glob", action="store_true", help="Use glob to search for input files")
parser.add_argument("-o", "--output_path", type=str, required=True, help="The output path for the merged json file")
args = parser.parse_args()

# Get the file names (glob for patterns if using glob)
files = [p for f in args.files for p in glob.glob(f)] if args.use_glob else args.files

# Check to make sure files list isn't empty
if len(files) == 0 and args.use_glob:
    print("Error: No files matched glob pattern(s)")
    exit(0)

# Read in all of the json files
print("Reading json files...")
jsons = []
for fname in files:
    with open(fname) as f:
        jsons.append(json.load(f))

# Merge the results
results = [item for d in jsons for item in d]

# Write to the output file
print(f"Writing to output file path {args.output_path}...")
with open(args.output_path, "w") as f:
    json.dump(results, f)
