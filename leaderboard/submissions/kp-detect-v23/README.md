# kp-detect-v23

Ensemble of a transformer-based semantic classifier (DeBERTa-v3-base, 768-D mean-pooled embeddings + logistic regression) and an attack-feature gradient boosting detector (41-D engineered features, 5-seed ensemble, global temperature calibration). Routing by predicted attack type: semantic classifier weighted more heavily for paraphrase attacks, feature-based detector used alone for character-manipulation attacks.

**Contact:** kelsamadicy@gmail.com
