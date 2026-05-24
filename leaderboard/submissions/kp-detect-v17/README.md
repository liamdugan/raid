# kp-detect-v17 submission

A hybrid statistical + Unicode-aware AI-generated text detector. 30-D
engineered text features → calibrated gradient-boosted classifier, with a
Unicode-normalization pre-processing step for adversarially-modified inputs
(zero-width spaces, mathematical-alphanumeric homoglyphs).

- `metadata.json` — author + release date + contact (template fields)
- `predictions.json` — one `{id, score}` per RAID test row (score is P(AI))
