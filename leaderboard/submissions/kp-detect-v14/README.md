Text classifier submission. Predictions for the RAID held-out test set.

## predictions.json

JSON array of `{"id": <uuid>, "score": <float in [0,1]>}` objects, one per
RAID test row. `score` is the probability that the row is machine-generated;
higher means more likely machine-generated.

## metadata.json

Three fields per the RAID submission spec: `date_released`, `detector_name`,
`contact_info`.
