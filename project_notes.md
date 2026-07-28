# SatCHM Project Notes

This document contains important project context, design decisions, and implementation notes for the SatCHM (Satellite Canopy Height Model) project.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Architecture](#environment-architecture)
3. [Recent Changes](#recent-changes)
4. [Design Decisions](#design-decisions)
5. [Known Issues & Workarounds](#known-issues--workarounds)
6. [Development Notes](#development-notes)

---

## Project Overview

SatCHM is a machine learning pipeline for generating Canopy Height Models (CHM) from satellite imagery using a multi-scale neural network. The workflow consists of:

1. **Data Preparation** - Download and process LiDAR, DEM, and satellite imagery
2. **Training** - Train a PyTorch Lightning model on paired satellite/LiDAR data
3. **Inference** - Generate CHM predictions for new areas using trained model

**Key Technologies:**
- PyTorch Lightning for model training
- R + lidR for LiDAR processing
- PDAL for point cloud manipulation
- GDAL/Rasterio for raster operations
- GeoPandas for vector operations

---

## Environment Architecture

### Two-Environment Design (2026-07-24)

The project uses a **dual-environment architecture** to separate concerns and avoid dependency conflicts:

#### CPU Environment (`conda activate SatCHMenv`)
**Purpose:** Data preparation and preprocessing  
**Location:** Conda environment  
**Key Dependencies:**
- R (4.5) + lidR, terra, sf packages
- PDAL (2.10.0) + python-pdal for LiDAR processing
- GDAL, rasterio, geopandas for geospatial operations
- Full scientific Python stack (numpy, pandas, scipy, etc.)

**Used For:**
- `prepTrainInputs/main1.py` - LiDAR download and processing
- `prepTrainInputs/main2.py` - Satellite imagery processing
- Any operations requiring R or PDAL

#### GPU Environment (`source ~/.venvs/msnet-cu121/bin/activate`)
**Purpose:** Model training and inference  
**Location:** Python virtual environment  
**Key Dependencies:**
- PyTorch 2.3.0 + CUDA 12.1
- PyTorch Lightning 2.2.5
- Minimal geospatial stack (rasterio, geopandas, pyproj)
- Basic scientific Python (numpy, pandas)

**Used For:**
- `ms_net/train.py` - Model training
- `infer/main.py` - Inference (CHM prediction)

### Why Two Environments?

1. **Dependency Isolation:** R packages and PDAL have complex C++ dependencies that conflict with PyTorch's CUDA requirements
2. **Performance:** GPU environment is lightweight and optimized for PyTorch
3. **Flexibility:** Can run data prep on CPU servers, training/inference on GPU servers
4. **Maintainability:** Easier to debug and update when concerns are separated

### PDAL Optional Import (2026-07-24)

**Problem:** PDAL is required in `prepTrainInputs/utils.py` but cannot be easily installed in the GPU environment (requires C++ compilation, conda-specific builds).

**Solution:** Made PDAL an optional dependency:
```python
try:
    import pdal
    PDAL_AVAILABLE = True
except ImportError:
    PDAL_AVAILABLE = False
    pdal = None
```

**Impact:**
- ✅ GPU environment can import `utils.py` without PDAL
- ✅ Inference works on GPU servers without environment switching
- ✅ CPU environment still has full PDAL functionality
- ❌ Calling `laz()` without PDAL raises clear error message

**Files Modified:**
- `prepTrainInputs/utils.py` (lines 37-42, 1910-1915)

---

## Recent Changes

### 2026-07-24: Environment Architecture Improvements & GDAL Fix

#### PDAL Optional Import
- **Modified:** `prepTrainInputs/utils.py` to make PDAL optional (lines 37-42, 1910-1915)
- **Rationale:** Users were unable to run inference on GPU servers due to PDAL import errors. The optional import allows inference to run in the lightweight GPU environment while preserving full functionality in the CPU environment.

#### GDAL Version Conflict Fix
- **Issue:** `cropTif()` function failed during inference with "ERROR 4: not recognized as supported file format"
- **Root Cause:** System GDAL (used by `gdal_translate` subprocess) couldn't read files created by rasterio's GDAL
- **Solution:** Rewrote `cropTif()` to use rasterio directly instead of subprocess calls (lines 1013-1067)
- **Impact:** Inference now completes successfully on GPU servers without GDAL version conflicts

#### Documentation
- **Added:** Two-environment setup documentation
- **Added:** `.clinerules` file for AI assistant guidelines
- **Updated:** `command_reference.md` with dual-environment workflow
- **Merged:** `daily_notes.md` into `project_notes.md`

**Commits:**
- Modified `prepTrainInputs/utils.py` - Made PDAL optional import
- Modified `prepTrainInputs/utils.py` - Fixed cropTif GDAL version conflict
- Updated `command_reference.md` - Added dual-environment workflow
- Created `project_notes.md` - Comprehensive project documentation

### 2026-07-23: Git Workflow and Security Improvements

#### Bug Fix: IndexError in createLidarData
- **Issue:** `IndexError: list index out of range` when running `main1.py`
- **Location:** `prepTrainInputs/utils.py` (formerly main1Utils.py)
- **Root Cause:** Regex pattern trying to access `[-1]` on empty list when lidar URLs didn't contain year information
- **Fix Applied:** 
  - Changed from one-liner to loop-based approach
  - Added check for empty matches before accessing list
  - Added warning messages for URLs without years
  - Added validation to ensure at least one year was found
  - Raises clear error if no years found in any URLs
- **Status:** ✅ Fixed and tested on ryan_dev branch

#### Security: Protected .env Files
- **Issue:** `.env_lm`, `.env_qm`, `.env_ws` were being tracked by git on ryan_dev branch
- **Risk:** API keys and sensitive configuration could be committed/pushed
- **Actions Taken:**
  1. Updated `.gitignore` to include `.env*` pattern (catches all .env files)
  2. Removed files from git tracking: `git rm --cached .env_lm .env_qm .env_ws`
  3. Committed changes to protect all .env files
- **Result:** Files remain in working directory but are no longer tracked by git
- **Status:** ✅ Complete - .env files are now safe

#### Git Workflow: Synchronized .gitignore Between Branches
- **Issue:** `.gitignore` was different between `ryan_dev` and `v1.0.0` branches
- **Risk:** Personal files would not be protected when on v1.0.0 branch
- **Actions Taken:**
  1. Switched to `v1.0.0` branch
  2. Copied `.gitignore` from `ryan_dev`: `git checkout ryan_dev -- .gitignore`
  3. Committed to `v1.0.0`: "Update .gitignore to protect .env*, command_reference.md, and daily_notes.md"
  4. Verified both branches have identical `.gitignore` files
- **Result:** Both branches now protect the same files - safe to switch between branches
- **Status:** ✅ Complete

#### Documentation
- **Created:** `command_reference.md` - Comprehensive command reference guide
- **Created:** `daily_notes.md` - Daily development tracking (now merged into this file)
- **Protection:** Added to `.gitignore` - stays on ryan_dev only

**Commits:**
- `1418ac4` - "Stop tracking .env_* files and update .gitignore to protect all .env files" (ryan_dev)
- `b0727d9` - "Add command_reference.md to gitignore" (ryan_dev)
- `41d4362` - "Add daily_notes.md to gitignore" (ryan_dev)
- `c12ebc8` - "Update .gitignore to protect .env*, command_reference.md, and daily_notes.md" (v1.0.0)

---

## Design Decisions

### Tile-Based Processing
**Decision:** Process imagery and LiDAR in 256m × 256m tiles with 16m halo  
**Rationale:**
- Manageable memory footprint
- Enables parallel processing
- Halo prevents edge artifacts in predictions
- Standard size simplifies data management

### Training Data Alignment
**Decision:** Automatically compute temporal alignment year from LiDAR metadata  
**Rationale:**
- LiDAR and satellite imagery must be temporally aligned
- Automated extraction from USGS 3DEP URLs
- Fallback to 2020 if year cannot be determined

### Model Checkpointing
**Decision:** Save checkpoints every 10 epochs during training  
**Rationale:**
- Balance between disk space and recovery granularity
- Allows resuming from recent checkpoint if training interrupted
- Enables comparison of model performance over time

### Inference Shape Buffering
**Decision:** Buffer inference areas by 512m during DEM download  
**Rationale:**
- Ensures complete coverage at tile edges
- Accounts for potential alignment issues
- Small overhead for improved reliability

---

## Known Issues & Workarounds

### PDAL Installation on GPU Servers
**Issue:** PDAL requires conda and cannot be installed via pip  
**Status:** Resolved via optional import (2026-07-24)  
**Workaround:** Use CPU environment for data prep, GPU environment for training/inference

### IndexError in createLidarData
**Issue:** Some USGS 3DEP URLs don't contain year information  
**Status:** Fixed in ryan_dev branch  
**Workaround:** Fallback to default year (2020) if extraction fails

### Memory Usage During Training
**Issue:** Large batch sizes can cause OOM errors  
**Status:** Expected behavior  
**Workaround:** Reduce batch size in `train.py` or use fewer training tiles

### Edge Tiles with NoData
**Issue:** Tiles at dataset boundaries may contain NoData pixels  
**Status:** Expected behavior  
**Workaround:** Tiles with NoData are automatically rejected during processing

---

## Development Notes

### Code Organization
```
SatCHM/
├── prepTrainInputs/     # Data preparation scripts
│   ├── main1.py         # LiDAR and DEM processing
│   ├── main2.py         # Satellite imagery processing
│   ├── utils.py         # Shared utility functions
│   └── R_chm_cli.R      # R interface for lidR
├── ms_net/              # Model training
│   ├── train.py         # Training script
│   ├── network_2D_lightning.py  # Model architecture
│   └── ...
├── infer/               # Inference
│   └── main.py          # Inference script
└── setup/               # Environment setup
    └── SatCHMenv.yml    # Conda environment spec
```

### Testing Checklist
When making changes, test:
- [ ] Data prep runs in CPU environment
- [ ] Training runs in GPU environment
- [ ] Inference runs in GPU environment
- [ ] No PDAL import errors in GPU environment
- [ ] Git ignores sensitive files (.env, .clinerules)

### Git Branch Strategy
- `main` - Stable releases
- `v1.0.0` - Version-specific branches
- `ryan_dev` - Active development branch
- Feature branches as needed

### Performance Considerations
- LiDAR download is the bottleneck in data prep (1-2 hours)
- Training time depends on GPU and number of tiles
- Inference is relatively fast (minutes for typical areas)
- Disk space: ~10-50 GB per site depending on imagery

---

## Future Improvements

### Potential Enhancements
1. **Automated imagery ordering** - API integration with Maxar/Vantor
2. **Multi-GPU training** - Distributed training for larger datasets
3. **Cloud deployment** - Containerized workflow for cloud platforms
4. **Uncertainty quantification** - Prediction confidence estimates
5. **Real-time inference** - Streaming inference for large areas

### Technical Debt
- Some hardcoded paths in scripts (should use config files)
- Limited error handling in some utility functions
- Could benefit from more comprehensive unit tests
- Documentation could be expanded with more examples

### Critical Issue: Model Weight Management (Discovered 2026-07-24)

**Problem:** The current training/inference workflow has a critical flaw in how model weights are managed across multiple sites.

**Current Behavior:**
- `ms_net/train.py` creates sequential version folders (`version_0`, `version_1`, `version_2`, etc.)
- `infer/main.py` automatically selects the **highest version number** for inference
- No tracking of which version was trained on which site

**Critical Issues:**
1. **Model Mismatch Across Sites** 🔴
   - Training Site A creates `version_7` with Site A data
   - Training Site B creates `version_8` with Site B data
   - Running inference on Site A uses `version_8` (trained on Site B!)
   - **Result:** Wrong model applied to wrong site = poor predictions

2. **Accidental Model Overwriting** 🔴
   - Training a new site overwrites your "best" model
   - No way to preserve site-specific models
   - Can't easily revert to previous models

3. **No Model Tracking** 🔴
   - No metadata linking versions to sites/training dates
   - No performance metrics stored with models
   - Difficult to identify which model to use for which site

**Potential Solutions:**

**Option 1: Site-Specific Model Directories** (Recommended)
```python
# In train.py, modify logger:
logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name=f"{site}_model",  # e.g., "ws_model", "lm_model"
    version=None  # auto-increment within site
)
```
- Pros: Clean separation, easy to manage
- Cons: Requires code modification

**Option 2: Manual Weight Selection**
```python
# In infer/main.py, uncomment line 88 and specify:
pathToWeights = '/path/to/specific/site/epoch-epoch=999.ckpt'
```
- Pros: Quick fix, no training code changes
- Cons: Manual management, error-prone

**Option 3: Metadata Tracking System**
- Save `model_metadata.json` with each version
- Track: site, training date, num_tiles, performance metrics
- Modify inference to select weights based on matching site
- Pros: Most robust, enables model comparison
- Cons: Requires significant code changes

**Immediate Recommendation:**
1. Back up current best models before training new sites
2. Document which version corresponds to which site
3. Consider implementing Option 1 for future training runs
4. Discuss with development team for long-term solution

**Status:** 🔴 Critical - Needs developer review and decision on implementation approach

---

## Multi-Site Workflow Issues (Discovered 2026-07-24)

### Current Workflow

Users currently manage multiple sites by:
1. Creating separate `.env_{site}` files for each site
2. Manually copying the appropriate file to `.env` before running scripts
3. Running scripts which read site name from `.env`

**Problem:** This manual switching is error-prone and can cause issues when running scripts out of sequence or with the wrong `.env` file active.

---

### 🔴 CRITICAL ISSUES

#### 1. Model Weight Management (See Above)
- Inference always uses highest version number
- No tracking of which model was trained on which site
- **Impact:** Wrong model applied to wrong site = poor predictions

#### 2. Shared Output Directories
- **Location:** `ms_net/train.py` (lines 172-180)
- **Problem:** All training runs write to same `lightning_logs/` directory
- **Impact:** 
  - Training Site B overwrites Site A's checkpoints
  - No way to preserve site-specific models
  - Version numbers increment globally, not per-site
- **Example:**
  ```
  lightning_logs/
    version_0/  # Could be from any site
    version_1/  # Could be from any site
    version_7/  # Currently selected for ALL inference
  ```

---

### ⚠️ MODERATE ISSUES

#### 3. Site-Specific Data Paths
- **Locations:** 
  - `main1.py` line 34: `site_data_path = f'{site}_data'`
  - `main2.py` line 19: `site_data_path = f'{site}_data'`
  - `train.py` line 208: `data_path = f'{site}_data'`
  - `infer/main.py` line 42: `inf_data_path = f'{site}_INF_data'`
- **Current Behavior:** ✅ Each site has separate data directory
- **Potential Issue:** If wrong `.env` is active, scripts look for wrong site's data
- **Risk:** Moderate - Usually fails with "file not found" (safe failure)

#### 4. Train/Val/Test Lists
- **Location:** `train.py` lines 59-60
- **Files:** `{site}_trainlist.txt`, `{site}_vallist.txt`
- **Behavior:** Lists are site-specific and stored in site's data directory
- **Risk:** LOW - Fails fast if wrong `.env` (file not found)
- **Safety:** ✅ Won't accidentally train on wrong site's data

#### 5. Metadata and Intermediate Files
- **Locations:**
  - `main1.py` lines 41-44: `downloads/{site}/wvimgTrain`, `wvimgInf`
  - `main2.py` line 35: `downloads/{site}/metadata/DGTilesMetadata.json`
  - `main2.py` line 48: `downloads/{site}/trainShape/{site}_trainAnchors.csv`
- **Current Behavior:** ✅ Site-specific subdirectories
- **Potential Issue:** Running with wrong `.env` creates files in wrong site's directory
- **Risk:** MODERATE - Creates confusion but doesn't corrupt existing data

---

### ✅ SAFE BEHAVIORS

#### 6. Inference Shape Processing
- **Location:** `infer/main.py` lines 44-46
- **Behavior:** Uses `.env` to determine site and paths
- **Safety:** ✅ Will fail if files don't exist (safe failure mode)

#### 7. Data Preparation Scripts
- **Locations:** `main1.py`, `main2.py`
- **Behavior:** All paths derived from `.env` site variable
- **Safety:** ✅ Creates site-specific directories, won't overwrite other sites

---

### 🎯 RECOMMENDATIONS

#### Immediate Actions (Quick Fixes)

1. **Add Site Validation to All Scripts**
   ```python
   # At start of each script
   expected_site = os.getenv('site')
   if not os.path.exists(f'{expected_site}_data'):
       raise ValueError(f"Data directory for site '{expected_site}' not found. Wrong .env file?")
   print(f"✓ Running for site: {expected_site}")
   ```

2. **Document Model-Site Mapping**
   - Create `model_registry.txt` to track which version corresponds to which site
   - Update after each training run

3. **Create Workflow Checklist**
   - Document which `.env` to use for each operation
   - Add verification steps before running scripts

#### Long-Term Solutions (Recommended)

**Option A: Command-Line Site Flag** (RECOMMENDED)
```bash
# Instead of switching .env files:
python prepTrainInputs/main1.py --site ws
python prepTrainInputs/main2.py --site ws
python ms_net/train.py --site ws
python infer/main.py --site ws
```

**Benefits:**
- No manual file switching
- Explicit site specification in command
- Can't forget which site you're working on
- Easy to script/automate
- Command history shows which site was used

**Implementation:**
- Add `argparse` to each script
- Read site from command line first, fall back to `.env`
- Update all path constructions to use the site variable
- Modify model saving to include site name

**Option B: Config File System**
```bash
python train.py --config configs/wesner_springs.yaml
```

**Benefits:**
- All site parameters in one file
- Version controlled
- Easy to share/reproduce
- Can include site-specific hyperparameters

**Option C: Site Registry System**
Create `sites.json` to track all sites:
```json
{
  "ws": {
    "full_name": "Wesner Springs",
    "epsg": 32613,
    "trained": true,
    "model_version": 7,
    "training_date": "2026-07-20",
    "num_tiles": 1000
  },
  "lm": {
    "full_name": "Lookout Mountain",
    "epsg": 32613,
    "trained": true,
    "model_version": 5,
    "training_date": "2026-07-15",
    "num_tiles": 800
  }
}
```

---

### 📋 WORKFLOW SAFETY CHECKLIST

**Before running ANY script:**
- [ ] Verify correct `.env` file is active: `cat .env | grep site=`
- [ ] Check expected data directory exists: `ls -d {site}_data`
- [ ] For training: Verify train/val lists exist in data directory
- [ ] For inference: Verify correct model weights will be used
- [ ] Document which version corresponds to which site

**After training:**
- [ ] Record in `model_registry.txt`: `version_X trained on {site} on {date}`
- [ ] Back up the checkpoint if it's your best model
- [ ] Update site registry with new version number

**Before inference:**
- [ ] Verify model version matches the site you're inferring on
- [ ] Check that `.env` site matches inference area

---

### 💡 PROPOSED: `--site` Flag Implementation

**Goal:** Eliminate manual `.env` switching by adding `--site` flag to all scripts.

**Example Usage:**
```bash
# Data preparation
python prepTrainInputs/main1.py --site ws
python prepTrainInputs/main2.py --site ws

# Training
python ms_net/train.py --site ws

# Inference
python infer/main.py --site ws
```

**Implementation Plan:**

1. **Add argparse to each script:**
   ```python
   import argparse
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--site', type=str, required=True, 
                       help='Site code (e.g., ws, lm, qm)')
   args = parser.parse_args()
   
   # Override .env with command line
   site = args.site
   ```

2. **Modify model saving to include site:**
   ```python
   # In train.py
   logger = TensorBoardLogger(
       save_dir="lightning_logs",
       name=f"{site}_model",
       version=None
   )
   ```

3. **Update inference to match site:**
   ```python
   # In infer/main.py
   weightsRoot = os.path.join(project_path, 'SatCHM', 'ms_net', 'lightning_logs')
   site_model_dir = os.path.join(weightsRoot, f"{site}_model")
   # Select latest version from THIS site's models only
   ```

4. **Add validation:**
   ```python
   # Check that site data exists
   if not os.path.exists(f'{site}_data'):
       raise ValueError(f"No data found for site '{site}'. Run data prep first.")
   ```

**Benefits:**
- ✅ No more manual `.env` switching
- ✅ Explicit site in every command
- ✅ Site-specific model directories
- ✅ Can't accidentally use wrong model
- ✅ Easy to automate with shell scripts
- ✅ Command history shows which site was used

**Migration Path:**
1. Implement `--site` flag (keep `.env` as fallback)
2. Test with one site
3. Migrate all sites
4. Eventually deprecate `.env` site variable

**Status:** ✅ IMPLEMENTED (2026-07-24) - Ready for Testing

**Implementation Details:**
- All 4 scripts updated with `--site` flag support
- Site-specific model directories created automatically
- Backward compatible with existing `.env` workflow
- See `SITE_FLAG_IMPLEMENTATION.md` for complete guide

**Testing Status:** 🧪 Pending user testing on multiple sites

---

## Contact & Resources

**Repository:** https://github.com/lanl/CHM-MS-net-Canopy-Height-Model  
**Documentation:** See README.md and howToRunSatCHM.md  
**Issues:** Report via GitHub Issues or contact maintainers

---

*Last updated: 2026-07-24*
