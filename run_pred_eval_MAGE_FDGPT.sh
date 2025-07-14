#!/bin/bash
#
#SBATCH --job-name=mage_fdgpt
#SBATCH --output=/nlp/data/ldugan/raid/logs/R-%x.%j.out
#SBATCH --constraint=48GBgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00

# Run predictions and evaluation on MAGE
python /nlp/data/ldugan/raid/run_pred_eval.py -m fastdetectgpt -d /nlp/data/ldugan/moe-detectors/joint_training/mage_test_processed.csv -p /nlp/data/ldugan/raid/outputs/fdgpt_mage_predictions.json -e /nlp/data/ldugan/raid/outputs/fdgpt_mage_eval_result.json