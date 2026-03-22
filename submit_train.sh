#!/bin/bash
#SBATCH --job-name=ravss_train_lrs2
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@iitb.ac.in

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your HPC)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate environment
source hpc_venv/bin/activate

# Navigate to project directory
cd AVSS

# Start training
echo "Starting training on $(date)"
python train_lrs2_2mix.py run config/lrs2_2mix_train.yaml \
    --nproc_per_node 4 \
    --batch_size 4 \
    --epochs 100

echo "Training completed on $(date)"

# Optional: Copy results to backup
# scp -r experiments_LRS2 username@local_machine:/path/to/backup/
