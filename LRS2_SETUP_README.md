# LRS2-2Mix Testing Setup - Complete ✅

## What Has Been Created

### 1. **Dataset Loader** (`dataset_lrs2_2mix.py`)
- Loads audio mixture, clean sources, and visual features from LRS2-2Mix
- Automatically handles:
  - Audio from `audio/wav16k/min/{tr|cv|tt}/mix/`
  - Clean sources from `audio/wav16k/min/{tr|cv|tt}/s1/` and `s2/`
  - Visual features from `mouths/*.npz` (face crops)
- Created 3000 test samples with 1500 batches

### 2. **Config File** (`config/lrs2_2mix_test.yaml`)
- Pre-configured for 2-speaker separation
- Points to LRS2-2Mix dataset paths
- SI-SNR loss for evaluation
- PESQ metric for quality assessment

### 3. **Test Script** (`run_lrs2_2mix_test.py`)
- Adapted from original `run_ravss_test.py`
- Handles distributed evaluation across GPUs
- Logs metrics to output directory
- Includes error handling and progress reporting

### 4. **Validation Script** (`validate_lrs2_dataset.py`)
- Checks dataset structure and files
- Generates statistics
- Useful for debugging

---

## Quick Test Command

```bash
cd /Users/sumanth/Desktop/DDP/RAVSS/ravss_code

# Activate venv first
source .venv/bin/activate

# Run test on LRS2-2Mix
python run_lrs2_2mix_test.py run config/lrs2_2mix_test.yaml \
  --test_cdp path/to/your/checkpoint.pth
```

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Test Samples** | 3000 |
| **Batch Size** | 2 |
| **Total Batches** | 1500 |
| **Audio Shape** | [batch, 32000 samples @ 16kHz = 2 sec] |
| **Sources Shape** | [batch, 64000 samples = 2 speakers] |
| **Visual Shape** | [batch, frames, height, width] |

---

## Important Notes

1. **Clean Sources Available** ✅
   - LRS2-2Mix provides isolated speaker audio (s1/ and s2/)
   - Perfect for SI-SNR loss computation

2. **Visual Features**
   - Mouth crops from `.npz` files
   - Shape varies based on video resolution
   - Model should handle variable dimensions

3. **Configuration Parameters to Override**
   ```bash
   python run_lrs2_2mix_test.py run config/lrs2_2mix_test.yaml \
     --test_cdp checkpoint.pth \
     --batch_size 4 \
     --num_workers 2 \
     --nproc_per_node 2
   ```

4. **Output**
   - Metrics saved in `experiments_LRS2/` folder
   - Log file: `test.log`
   - SI-SNR loss and PESQ scores per batch

---

## Directory Structure Created

```
ravss_code/
├── dataset_lrs2_2mix.py          ← NEW: Dataset loader
├── config/
│   └── lrs2_2mix_test.yaml       ← NEW: Test config
├── run_lrs2_2mix_test.py         ← NEW: Test script
├── validate_lrs2_dataset.py       ← NEW: Validation utility
└── lrs2_rebuild/                 ← Your LRS2-2Mix dataset
    ├── audio/wav16k/min/
    │   ├── tr/  (train: 20779 samples)
    │   ├── tt/  (test: 3000 samples)
    │   └── cv/  (dev: 492 samples)
    ├── mouths/
    ├── faces/
    └── train.txt, test.txt, dev.txt
```

---

## Next Steps

1. **Add Your Checkpoint**: Update `test_cdp` in config with your model weights
2. **Run Test**: Execute the test command above
3. **Check Results**: Look in `experiments_LRS2/*/test.log` for metrics

Good luck! 🚀
