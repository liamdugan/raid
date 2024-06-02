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
        f" committed detailed results of this detector's performance on the test set to this PR.\n"
    )

    # report aggregate scores
    score_agg = results["score_agg"]["all"]
    if "_note" in score_agg:
        print(f"\n> [!WARNING]\n> {score_agg['_note']}")
    else:
        print(
            "On the RAID dataset as a whole (aggregated across all generation models, domains, decoding strategies,"
            " repetition penalties, and adversarial attacks), it achieved an accuracy of "
            f"**{score_agg['accuracy']:.2%}**."
        )

    score_agg_no_adversarial = results["score_agg"]["no_adversarial"]
    if "_note" in score_agg_no_adversarial:
        print(f"\n> [!WARNING]\n> {score_agg_no_adversarial['_note']}")
    else:
        print(
            f"Without adversarial attacks, it achieved an accuracy of **{score_agg_no_adversarial['accuracy']:.2%}**."
        )

print(
    "If all looks well, a maintainer will come by soon to merge this PR and your entry/entries will appear on the"
    " leaderboard. If you need to make any changes, feel free to push new commits to this PR. Thanks for submitting to"
    " RAID!"
)
