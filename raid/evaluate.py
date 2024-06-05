import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def load_detection_result(df, results):
    # Load the dataframe and read in the scores
    scores_df = pd.DataFrame.from_records(results)

    # Merge dataframes based on the id and validate that ids are unique
    return df.join(scores_df.set_index("id"), on="id", validate="one_to_one")


def compute_fpr(y_scores, threshold):
    y_pred = [1 if y >= threshold else 0 for y in y_scores]
    y_true = [0] * len(y_pred)
    return 1 - accuracy_score(y_true, y_pred)


# Search to find threshold for FPR
def find_threshold(df, target_fpr, epsilon):
    iteration = 1
    prev_dist = None
    step_size = 0.5
    y_scores = df[(df["model"] == "human") & (df["attack"] == "none") & (df["score"].notnull())]["score"].tolist()
    threshold = sum(y_scores) / len(y_scores)  # initialize threshold to mean of y_scores
    sign = lambda x: -1 if x < 0 else 1

    # Initialize the list of all found thresholds and FPRs
    found_threshold_list = []
    while abs((fpr := compute_fpr(y_scores, threshold)) - target_fpr) > epsilon:
        # Save the computed values to the found_threshold_list
        found_threshold_list.append((threshold, fpr))

        # Increment the iteration count and compute distance
        iteration += 1
        dist = target_fpr - fpr

        # If dist and prev_dist are different signs then swap
        # sign of step size and cut in half
        if prev_dist and sign(dist) != sign(prev_dist):
            step_size *= -0.5
        # Otherwise if we're going the wrong direction, then just swap sign of step
        elif prev_dist and abs(dist) - abs(prev_dist) > 0.01:
            step_size *= -1

        # Step the threshold value and save prev_dist
        threshold += step_size
        prev_dist = target_fpr - fpr

        # Can't find the threshold, let's find the best one
        if iteration > 50:
            # Compute diffs for all thresholds found during search
            # (Exclude all thresholds for which the true fpr is 0)
            diffs = [(target_fpr - fpr, t) for t, fpr in found_threshold_list if fpr > 0.0]

            # If there are positive numbers in the list, pick threshold for smallest pos number
            # Otherwise pick the threshold for the negative diff value closest to 0
            if len(pos_diffs := [(d, t) for d, t in diffs if d >= 0]) > 0:
                threshold = min(pos_diffs)[1]
            else:
                threshold = max(diffs)[1]

            break

    return threshold, compute_fpr(y_scores, threshold)


def compute_thresholds(df, fpr=0.05, epsilon=0.0005):
    thresholds = {}
    true_fprs = {}
    for d in df.domain.unique():
        t, true_fpr = find_threshold(df[df["domain"] == d], fpr, epsilon)
        thresholds[d] = t
        true_fprs[d] = true_fpr
    return thresholds, true_fprs


def compute_scores(df, thresholds, remove_null=True):
    # Initialize the list of records for the scores
    scores = []

    # Filter out human data
    df = df[df["model"] != "human"]

    # For each domain, attack, model, and decoding strategy, filter the dataset
    for d in df.domain.unique().tolist() + ["all"]:
        dfd = df[df["domain"] == d] if d != "all" else df
        for a in df.attack.unique().tolist() + ["all"]:
            dfa = dfd[dfd["attack"] == a] if a != "all" else dfd
            for m in df.model.unique().tolist() + ["all"]:
                dfm = dfa[dfa["model"] == m] if m != "all" else dfa
                for s in df.decoding.unique().tolist() + ["all"]:
                    dfs = dfm[dfm["decoding"] == s] if s != "all" else dfm
                    for r in df.repetition_penalty.unique().tolist() + ["all"]:
                        df_filter = dfs[dfs["repetition_penalty"] == r] if r != "all" else dfs

                        # If no outputs for this split, continue
                        if (len(df_filter) == 0):
                            continue

                        # If we're removing null and there are null scores, continue
                        if remove_null and (len(df_filter[df_filter["score"].isnull()]) > 0):
                            continue

                        # Initialize predictions
                        preds = []

                        # For each domain in df_filter
                        for domain in df_filter.domain.unique():
                            # Filter the dataset to just that domain
                            df_domain = df_filter[df_filter["domain"] == domain]

                            # Select the domain-specific threshold to use for classification
                            t = thresholds[domain]

                            # Get the 0 to 1 scores for the detector
                            y_model = df_domain["score"].to_numpy()

                            # Threshold scores using the threshold for this detector
                            # Source: https://stackoverflow.com/a/45648782
                            y_pred = (y_model >= t).astype(int)

                            # Add the prediction array to the list of predictions
                            preds.append(y_pred)

                        # Concatenate the predictions together
                        y_pred = np.concatenate(preds, axis=0)
                        y_true = np.ones(len(y_pred))

                        # Calculate the true positives and false negatives
                        tp = y_pred.sum()
                        fn = len(y_pred) - tp

                        # Compute accuracy and add to scores
                        scores.append(
                            {
                                "domain": d,
                                "model": m,
                                "decoding": s,
                                "repetition_penalty": r,
                                "attack": a,
                                "tp": int(tp),
                                "fn": int(fn),
                                "accuracy": accuracy_score(y_true, y_pred),
                            }
                        )
    return scores


def run_evaluation(results, df, target_fpr=0.05, epsilon=0.0005):
    # Add detector outputs into a 'score' column
    df = load_detection_result(df, results)

    # Find thresholds per-domain for target FPR
    thresholds, fprs = compute_thresholds(df, target_fpr, epsilon)

    # Compute accuracy scores for each split of the data
    scores = compute_scores(df, thresholds)

    return {"scores": scores, "thresholds": thresholds, "fpr": fprs, "target_fpr": target_fpr}
