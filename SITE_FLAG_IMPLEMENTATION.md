# Site Flag Implementation Guide

## Overview

The `--site` flag has been successfully implemented across all SatCHM scripts to eliminate manual `.env` file switching and ensure site-specific model management.

**Date Implemented:** 2026-07-24  
**Branch:** ryan_dev  
**Status:** ✅ Complete - Ready for Testing

---

## What Changed?

### 🎯 Key Improvement: Site-Specific Model Directories

**Before:**
```
lightning_logs/
  version_0/  # Could be from ANY site
  version_1/  # Could be from ANY site
  version_7/  # Used for ALL inference
```

**After:**
```
lightning_logs/
  ws_model/
    version_0/  # Wesner Springs model
    version_1/  # Wesner Springs model
  lm_model/
    version_0/  # Lookout Mountain model
    version_1/  # Lookout Mountain model
  qm_model/
    version_0/  # Quemazon model
```

**Result:** Each site has its own model directory, and inference automatically uses the correct site's model!

---

## How to Use

### New Workflow (Recommended)

```bash
# Data Preparation
python prepTrainInputs/main1.py --site ws
python prepTrainInputs/main2.py --site ws

# Training
python ms_net/train.py --site ws

# Inference
python infer/main.py --site ws
```

### Old Workflow (Still Supported)

```bash
# Copy the appropriate .env file
cp .env_ws .env

# Run scripts (will use .env)
python prepTrainInputs/main1.py
python prepTrainInputs/main2.py
python ms_net/train.py
python infer/main.py
```

---

## Files Modified

### 1. `ms_net/train.py`
**Changes:**
- Added `argparse` for `--site` flag
- Added `TensorBoardLogger` with site-specific naming
- Models now saved to `lightning_logs/{site}_model/version_X/`
- Added validation to check if site data exists

**Key Code:**
```python
logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name=f"{site}_model",  # Site-specific directory!
    version=None
)
```

### 2. `infer/main.py`
**Changes:**
- Added `argparse` for `--site` flag
- Modified weight selection to look in site-specific directories
- Automatically selects latest model for the specified site
- Clear error message if no model exists for site

**Key Code:**
```python
site_model_dir = os.path.join(weightsRoot, f"{site}_model")
# Find latest version for THIS site only
weightsVersion = max(...)
```

### 3. `prepTrainInputs/main1.py`
**Changes:**
- Added `argparse` for `--site` flag
- Site can be specified via command line or `.env`
- Added validation and user feedback

### 4. `prepTrainInputs/main2.py`
**Changes:**
- Added `argparse` for `--site` flag
- Site can be specified via command line or `.env`
- Added validation and user feedback

---

## Command Line Options

### Training Script (`train.py`)
```bash
python ms_net/train.py --site SITE [--norm-const FLOAT]

Options:
  --site SITE           Site code (e.g., ws, lm, qm) [REQUIRED if not in .env]
  --norm-const FLOAT    Normalization constant (default: 46)
```

### Inference Script (`infer/main.py`)
```bash
python infer/main.py --site SITE [--norm-const FLOAT] [--feather-const INT]

Options:
  --site SITE           Site code (e.g., ws, lm, qm) [REQUIRED if not in .env]
  --norm-const FLOAT    Normalization constant (default: 46)
  --feather-const INT   Feathering constant for merging (default: 40)
```

### Data Prep Scripts (`main1.py`, `main2.py`)
```bash
python prepTrainInputs/main1.py --site SITE
python prepTrainInputs/main2.py --site SITE

Options:
  --site SITE           Site code (e.g., ws, lm, qm) [REQUIRED if not in .env]
```

---

## Benefits

### ✅ No More Manual File Switching
- No need to copy `.env_ws` to `.env` before each operation
- Explicit site specification in every command
- Can't forget which site you're working on

### ✅ Site-Specific Model Management
- Each site has its own model directory
- Training Site B won't overwrite Site A's models
- Inference automatically uses the correct site's model
- **Solves the critical model weight management issue!**

### ✅ Easy to Automate
```bash
# Train all sites in sequence
for site in ws lm qm; do
    echo "Training site: $site"
    python ms_net/train.py --site $site
done
```

### ✅ Clear Error Messages
```
ValueError: No model directory found for site 'ws' at: lightning_logs/ws_model
Train a model for this site first using: python ms_net/train.py --site ws
```

### ✅ Command History Shows Site
```bash
$ history | grep train
python ms_net/train.py --site ws
python ms_net/train.py --site lm
```

---

## Migration Guide

### For Existing Projects

If you have existing models in `lightning_logs/version_X/`:

**Option 1: Rename Existing Directories**
```bash
cd ms_net/lightning_logs
mkdir ws_model
mv version_7 ws_model/version_0
```

**Option 2: Start Fresh**
- Keep old models as backup
- Train new models with `--site` flag
- New models will be in site-specific directories

### For New Projects

Just use the `--site` flag from the start! Everything will be organized automatically.

---

## Testing Checklist

Before using in production, test:

- [ ] **Data Prep:** Run `main1.py --site test_site`
  - Verify creates `test_site_data/` directory
  - Check error handling if site data missing

- [ ] **Data Prep:** Run `main2.py --site test_site`
  - Verify processes correct site's data
  - Check creates site-specific lists

- [ ] **Training:** Run `train.py --site test_site`
  - Verify creates `lightning_logs/test_site_model/version_0/`
  - Check model checkpoints saved correctly
  - Verify can resume training

- [ ] **Inference:** Run `infer/main.py --site test_site`
  - Verify selects correct site's model
  - Check error if no model exists for site
  - Verify predictions use correct model

- [ ] **Multi-Site:** Train two different sites
  - Verify each has separate model directory
  - Check inference uses correct model for each site
  - Confirm no cross-contamination

---

## Troubleshooting

### Error: "Site must be specified via --site flag or in .env file"
**Solution:** Add `--site SITECODE` to your command or ensure `.env` has `site=SITECODE`

### Error: "Data directory not found"
**Solution:** Run data preparation first: `python prepTrainInputs/main1.py --site SITECODE`

### Error: "No model directory found for site 'X'"
**Solution:** Train a model first: `python ms_net/train.py --site X`

### Inference uses wrong model
**Check:** 
1. Are you specifying the correct `--site` flag?
2. Does `lightning_logs/{site}_model/` exist?
3. Run with explicit site: `python infer/main.py --site ws`

---

## Example Workflows

### Complete Workflow for New Site
```bash
# 1. Prepare .env file (or use existing .env_ws)
# 2. Run data preparation
python prepTrainInputs/main1.py --site ws
python prepTrainInputs/main2.py --site ws

# 3. Train model
python ms_net/train.py --site ws

# 4. Run inference
python infer/main.py --site ws
```

### Training Multiple Sites
```bash
# Train Wesner Springs
python ms_net/train.py --site ws

# Train Lookout Mountain
python ms_net/train.py --site lm

# Train Quemazon
python ms_net/train.py --site qm

# Each creates its own model directory!
```

### Inference on Multiple Sites
```bash
# Infer on Wesner Springs (uses ws_model)
python infer/main.py --site ws

# Infer on Lookout Mountain (uses lm_model)
python infer/main.py --site lm

# Each automatically uses the correct model!
```

---

## Technical Details

### Model Directory Structure
```
ms_net/lightning_logs/
├── ws_model/
│   ├── version_0/
│   │   ├── checkpoints/
│   │   │   ├── best-val-epoch=XX-step=XXXXXX.ckpt
│   │   │   └── epoch-epoch=XX.ckpt
│   │   ├── events.out.tfevents...
│   │   └── hparams.yaml
│   └── version_1/
│       └── ...
├── lm_model/
│   └── version_0/
│       └── ...
└── qm_model/
    └── version_0/
        └── ...
```

### Weight Selection Logic
1. Look for `lightning_logs/{site}_model/` directory
2. Find highest `version_X` number within that site's directory
3. Find highest `epoch=XX` checkpoint within that version
4. Use that checkpoint for inference

### Backward Compatibility
- Scripts still read `.env` if `--site` not provided
- Existing workflows continue to work
- Gradual migration is supported

---

## Future Enhancements

Potential improvements for future versions:

1. **Model Registry File**
   - Track training date, performance metrics per site
   - JSON file: `model_registry.json`

2. **Config Files**
   - Replace `.env` with YAML configs
   - `python train.py --config configs/wesner_springs.yaml`

3. **Model Comparison**
   - Compare performance across sites
   - Visualize training curves per site

4. **Automated Testing**
   - Unit tests for site-specific logic
   - Integration tests for full workflow

---

## Questions?

If you encounter issues or have questions:
1. Check this guide's Troubleshooting section
2. Review `project_notes.md` for additional context
3. Check git history: `git log --oneline -- ms_net/train.py`

---

**Last Updated:** 2026-07-24  
**Author:** Cline AI Assistant  
**Approved By:** Developer Team
