"""Given a list of newly-written results files as argv, output a nice little Markdown summary of them."""

import json
import os
import sys

RUN_LINK = os.getenv("RUN_LINK")

print(f"Eval run succeeded! Link to run: [link]({RUN_LINK})\n\nHere are the results of the submission(s):\n")

for fp in sys.argv[1:]:
    with open(fp) as f:
        results = json.load(f)

    print(
        f"### {results['detector_name']}\n*Release date: {results['date_released']}*\n\nI've"
        " committed detailed results of this detector's performance on the test set to this PR.\n"
    )

    # report aggregate scores
    score_agg = results["score_agg"]["all"]
    if "_note" in score_agg:
        print(f"\n> [!WARNING]\n> {score_agg['_note']}")
    elif len(null_fprs := [f"{float(fpr):.0%}" for fpr, acc in score_agg["accuracy"].items() if acc is None]) > 0:
        print(
            f"\n> [!WARNING]\n> Failed to find threshold values that achieve False Positive Rate(s): ({null_fprs}) on"
            " all domains. This submission will not appear in the main leaderboard for those FPR values; it will only"
            " be visible within the splits in which the target FPR was achieved."
        )

    if "accuracy" in score_agg and any(list(score_agg["accuracy"].values())):
        accuracies = [(fpr, acc) for fpr, acc in score_agg["accuracy"].items() if acc is not None]
        accuracy_print = " and ".join([f"**{acc['accuracy']:.2%}** at FPR={float(fpr):.0%}" for fpr, acc in accuracies])
        print(
            "On the RAID dataset as a whole (aggregated across all generation models, domains, decoding strategies,"
            " repetition penalties, and adversarial attacks), it achieved an AUROC of"
            f" {score_agg['auroc'] * 100.0:.2f} and a TPR of {accuracy_print}."
        )

    score_agg_no_adversarial = results["score_agg"]["no_adversarial"]
    if "_note" in score_agg_no_adversarial:
        print(f"\n> [!WARNING]\n> {score_agg_no_adversarial['_note']}")

    if "accuracy" in score_agg_no_adversarial and any(list(score_agg_no_adversarial["accuracy"].values())):
        accuracies = [(fpr, acc) for fpr, acc in score_agg_no_adversarial["accuracy"].items() if acc is not None]
        accuracy_print = " and ".join([f"**{acc['accuracy']:.2%}** at FPR={float(fpr):.0%}" for fpr, acc in accuracies])
        print(
            f"Without adversarial attacks, it achieved AUROC of {score_agg_no_adversarial['auroc'] * 100.0:.2f} and a"
            f" TPR of {accuracy_print}."
        )

print()
print(
    "If all looks well, a maintainer will come by soon to merge this PR and your entry/entries will appear on the"
    " leaderboard. If you need to make any changes, feel free to push new commits to this PR. Thanks for submitting to"
    " RAID!"
)
