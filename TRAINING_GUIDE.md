# RAVSS Training Quick Start Guide

## Files Created

```
train_lrs2_2mix.py              ← Training script
config/lrs2_2mix_train.yaml     ← Training config
submit_train.sh                 ← SLURM job submission
```

---

## Quick Commands

### **1. Test Setup (Before Full Training)**

```bash
# Single GPU, 1 epoch, small batch
python train_lrs2_2mix.py run config/lrs2_2mix_train.yaml \
  --epochs 1 \
  --batch_size 2 \
  --nproc_per_node 1
```

### **2. Train on Single GPU**

```bash
python train_lrs2_2mix.py run config/lrs2_2mix_train.yaml \
  --nproc_per_node 1 \
  --batch_size 4
```

### **3. Train on 4 GPUs (Multi-GPU)**

```bash
python train_lrs2_2mix.py run config/lrs2_2mix_train.yaml \
  --nproc_per_node 4 \
  --batch_size 4
```

### **4. Submit to SLURM (HPC)**

```bash
# Make script executable
chmod +x submit_train.sh

# Edit the script first to:
# 1. Uncomment module commands if needed
# 2. Change email address
# 3. Adjust GPU count and time

# Submit job
sbatch submit_train.sh

# Check status
squeue -u $USER

# Monitor output
tail -f logs/train_*.log

# Cancel if needed
scancel JOB_ID
```

### **5. Resume Training from Checkpoint**

```bash
python train_lrs2_2mix.py run config/lrs2_2mix_train.yaml \
  --resume_from experiments_LRS2/some_folder/checkpoint_*.pt
```

---

## Config Options

Edit `config/lrs2_2mix_train.yaml`:

| Option | Default | Description |
|--------|---------|-------------|
| `batch_size` | 4 | Batch size per GPU |
| `epochs` | 100 | Number of training epochs |
| `lr` | 0.0001 | Learning rate |
| `nproc_per_node` | 1 | GPUs per node |
| `num_workers` | 4 | Data loading workers |

---

## Troubleshooting

### **Out of Memory (OOM)**
- Reduce `batch_size` (try 2 or 1)
- Reduce `num_workers`

### **Slow Training**
- Increase `num_workers` (try 8)
- Increase `batch_size`
- Use `--nproc_per_node 4` for multi-GPU

### **Model Not Improving**
- Adjust `lr` (try 0.00005 or 0.0002)
- Train longer (increase `epochs`)
- Check data format

---

## Output

Training saves to: `experiments_LRS2/AV_test_RAVSS_lrs2_YYYYMMDD-HHMMSS/`

Files:
- `train.log` - Training logs
- `checkpoint_*.pt` - Best model checkpoints
- `best_*.pt` - Final best model

---

## Next Steps

1. ✅ Created training script
2. 📝 Edit `submit_train.sh` with your HPC details
3. 🚀 Run test with `--epochs 1`
4. 📊 Submit full training with `sbatch submit_train.sh`
5. 📈 Monitor with `tail -f logs/train_*.log`

**Good luck! 🎯**
