def run_detection(f, df):
    # Run the detector function on the dataset and put output in score column
    df["score"] = f(df["generation"])

    # Convert scores and ids to dict in 'records' format for seralization
    # e.g. [{'id':'...', 'score':0}, {'id':'...', 'score':1}, ...]
    results = df[["id", "score"]].to_dict(orient="records")

    return results
