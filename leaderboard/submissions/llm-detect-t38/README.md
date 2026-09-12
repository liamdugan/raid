# LLM Detect T38

Predictions for all 672,000 rows of the official RAID hidden-label test set.
This is an experimental multilingual adaptation in the Desklib/Oculus model lineage.
The checkpoint was fixed using existing development evaluations before RAID inference.

The full text is split using the detector service's paragraph and sentence-aware
chunking, with at most 512 tokens including special tokens in each chunk. Chunk
logits are averaged with content-token weights, then mapped through sigmoid.
No text normalization, attack-specific rules, domain-specific routing, test-label
access, or RAID-based threshold fitting is performed by this submission.
Official RAID evaluation determines its own domain-specific thresholds.

Inference ran remotely on two NVIDIA RTX 4090 GPUs with BF16 weights/autocast.
Each GPU processed a disjoint contiguous half of the test set; all predictions use T38.
After 497,024 committed predictions, inference added exact-shape CUDA graph replay,
constant relative-position caching and equivalent NumPy padding. Prior predictions
were retained. BF16 logits matched exactly in the recorded acceleration probe;
model weights, tokenizer, batching limits and aggregation were unchanged.
The graph cache was later limited to four short-sequence shapes to bound memory;
this scheduling change retained all 631,296 predictions committed at that point. Precision
probe results and artifact hashes are recorded in metadata.json. RAID official train
samples were used during earlier English replay training; this is not a zero-shot
submission. Upstream training overlap is not fully auditable.

Only predictions and metadata are submitted. Official results must be generated
by the RAID evaluation workflow.
