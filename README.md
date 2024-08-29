<p align="center">
  &emsp;&emsp;<a href="https://raid-bench.xyz"><img src="assets/logo.png" alt="RAID" width="400"></a><br />
  <a href="https://raid-bench.xyz"><b>https://raid-bench.xyz</b></a>
</p>
<p align="center">
   <b>Open Leaderboards. Trustworthy Evaluation. Robust AI Detection.</b>
</p>
<p align="center">
  <a href="https://github.com/liamdugan/raid/actions/workflows/lint.yml"><img src="https://img.shields.io/github/actions/workflow/status/liamdugan/raid/lint.yml?logo=githubactions&logoColor=white&label=Code Style%20" alt="Code Style" style="max-width: 100%;"></a>
  <a href="https://pypi.org/project/raid-bench/"><img src="https://badge.fury.io/py/raid-bench.svg"/></a>
  <a href="https://raid-bench.xyz/"><img src="https://img.shields.io/website.svg?down_color=red&down_message=offline&label=Leaderboard%20%26%20Website&up_message=online&url=https://raid-bench.xyz/"/></a>
  <br />
  <a href="https://github.com/liamdugan/raid/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://liamdugan.com/"><img src="https://img.shields.io/badge/NLP-NLP?labelColor=011F5b&color=990000&label=University%20of%20Pennsylvania"/></a>
  <a href="https://arxiv.org/abs/2405.07940"><img src="https://img.shields.io/badge/arXiv-2405.07940-b31b1b.svg"/></a>
</p>

# RAID: Robust AI Detection

This repository contains the code for the ACL 2024 paper [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://arxiv.org/abs/2405.07940). In our paper we introduce the RAID dataset and use it to show that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models.

## Usage

### Pypi package (recommended)

If you want to run RAID on a new detector, we recommend using our pypi package. To install first run `pip install raid-bench` and then use the `run_detection` and `run_evaluation` functions as follows: 

Example:
```py
from raid import run_detection, run_evaluation
from raid.utils import load_data

# Define your detector function
def my_detector(texts: list[str]) -> list[float]:
    pass

# Load the RAID dataset
train_df = load_data(split="train")

# Run your detector on the dataset
predictions = run_detection(my_detector, train_df)

# Run evaluation on your detector predictions
evaluation_result = run_evaluation(predictions, train_df)
```

### Installing from Source

If you want to run the detectors we have implemented or use our dataset generation code you should install from source. To do so first clone the repository. Then install in your virtual environment of choice

Conda:

```
conda create -n raid_env python=3.9.7
conda activate raid_env
pip install -r requirements.txt
```

venv:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Then, populate the `set_api_keys.sh` file with the API keys for your desired modules (OpenAI, Cohere, API detectors, etc.). After that, run `source set_api_keys.sh` to set the API key evironment variables.

To apply a detector to the dataset through our CLI run `detect_cli.py` and `evaluate_cli.py`. These wrap around the `run_detection` and `run_evaluation` functions from the pypi package. The options are listed below. See `detectors/detector.py` for a list of valid detector names.

```
$ python detect_cli.py -h
  -m, --model           The name of the detector model you wish to run
  -d, --data_path       The path to the csv file with the dataset
  -o, --output_path     The path to write the result JSON file
```

```
$ python evaluate_cli.py -h
  -r, --results_path    The path to the detection result JSON to evaluate
  -d, --data_path       The path to the csv file with the dataset
  -o, --output_path     The path to write the result JSON file
  -t, --target_fpr      The target FPR to evaluate at (Default: 0.05)
```

Example:
```
$ python detect_cli.py -m gltr -d train.csv -o gltr_predictions.json
$ python evaluate_cli.py -i gltr_predictions.json -d train.csv -o gltr_result.json
```

The output of `evaluate_cli.py` will be a JSON file containing the accuracy of the detector on each split of the RAID dataset at the target false positive rate as well as the thresholds found for the detector.

### Running custom detectors via CLI

If you would like to implement your own detector and still run it via the CLI, you must add it to `detectors/detector.py` so that it can be called via command line argument.

## Data

The main RAID dataset is partitioned into 90% train and 10% test set. It includes generations from the following 8 domains: NYT News Articles, IMDb movie reviews, Paper Abstracts, Poems, Reddit posts, Recipes, Book Summaries, and Wikipedia.

To download the RAID train and test sets manually, run
```
$ wget https://dataset.raid-bench.xyz/train.csv
$ wget https://dataset.raid-bench.xyz/test.csv
```

We also release an extra split of the dataset which consists of generations from three extra domains: Python Code, German News, and Czech News. To download the extra data run

```
$ wget https://dataset.raid-bench.xyz/extra.csv
```

All code used to generate the RAID dataset is located in `/generation`. This includes implementations of generators,
adversarial attacks, metrics, filtering criteria and other sanity checks and validations.

## Leaderboard Submission

To submit to the leaderboard, you must first get predictions for your detector on the test set. You can do so using either the pypi package or the CLI:

### Using Pypi
```py
import json

from raid import run_detection, run_evaluation
from raid.utils import load_data

# Define your detector function
def my_detector(texts: list[str]) -> list[float]:
    pass

# Load the RAID test data
test_df = load_data(split="test")

# Run your detector on the dataset
predictions = run_detection(my_detector, test_df)

with open('predictions.json') as f:
    json.dump(predictions, f)
```

### Using CLI

```
$ python detect_cli.py -m gltr -d test.csv -o predictions.json
```

After you have the `predictions.json` file you must then write a metadata file for your submission. Your metadata file should use the template found in
this repository at `leaderboard/template-metadata.json`.

Finally, fork this repository. Add your generation files to `leaderboard/submissions/YOUR-DETECTOR-NAME/predictions.json` and your metadata file to `leaderboard/submissions/YOUR-DETECTOR-NAME/metadata.json` and make a pull request to this repository.

Our GitHub bot will automatically run evaluations on the submitted predictions and commit the results to
`leaderboard/submissions/YOUR-DETECTOR-NAME/results.json`. If all looks well, a maintainer will merge the PR and your
model will appear on the leaderboards!

> [!NOTE]
> You may submit multiple detectors in a single PR - each detector should have its own directory.

## Citation

If you use our code or findings in your research, please cite us as:

```
@inproceedings{dugan-etal-2024-raid,
    title = "{RAID}: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors",
    author = "Dugan, Liam  and
      Hwang, Alyssa  and
      Trhl{\'\i}k, Filip  and
      Zhu, Andrew  and
      Ludan, Josh Magnus  and
      Xu, Hainiu  and
      Ippolito, Daphne  and
      Callison-Burch, Chris",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.674",
    pages = "12463--12492",
}
```

## Acknowledgements

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

<!-- ## Dev Notes

We use [Black](https://black.readthedocs.io/en/stable/) code style and [isort](https://pycqa.github.io/isort/) for code
cleanliness.

See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ for Python packaging instructions. Publishing
a new release on GitHub will automatically deploy to PyPI.

To bump the package version, use `hatch version [major|minor|patch]` or edit `raid/_version.py`. -->
