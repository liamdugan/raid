#!/bin/bash
#
#SBATCH --job-name=raid_conda
#SBATCH --output=/nlp/data/ldugan/raid/logs/R-%x.%j.out
#SBATCH --constraint=48GBgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00

# Run predictions and evaluation on RAID
python /nlp/data/ldugan/raid/run_pred_eval.py -m conda -d /nlp/data/ldugan/moe-detectors/joint_training/test_none_with_labels.csv -p /nlp/data/ldugan/raid/outputs/conda_raid_predictions.json -e /nlp/data/ldugan/raid/outputs/conda_raid_eval_result.json