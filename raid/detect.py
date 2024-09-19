def run_detection(f, df):
    # Make a copy of the IDs of the original dataframe to avoid editing in place
    scores_df = df[["id"]].copy()

    # Run the detector function on the dataset and put output in score column
    scores_df["score"] = f(df["generation"].tolist())

    # Convert scores and ids to dict in 'records' format for seralization
    # e.g. [{'id':'...', 'score':0}, {'id':'...', 'score':1}, ...]
    results = scores_df[["id", "score"]].to_dict(orient="records")

    return results
