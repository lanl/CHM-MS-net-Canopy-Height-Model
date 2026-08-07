# sites.json Configuration Guide

## Overview

The `sites.json` file provides a centralized configuration system for managing multiple sites in SatCHM. This eliminates the need to manually copy `.env` files when switching between sites.

## Benefits

✅ **No more manual `.env` switching** - Just use `--site` flag  
✅ **Centralized configuration** - All sites in one file  
✅ **Easy to add new sites** - Just add a new entry  
✅ **Version controlled** - Track site configurations (except API keys)  
✅ **Backward compatible** - `.env` files still work as fallback  

---

## File Structure

### Location
```
/project/es14-stor/canopy/SatCHM/sites.json
```

### Format
```json
{
  "site_code": {
    "full_name": "Full Site Name",
    "epsg": 32613,
    "inference_shape": "site_area_32613.geojson",
    "openTopoAPIkey": "your_api_key_here",
    "notes": "Optional notes",
    "customTrainShpPath": "/optional/path/to/custom/shape.geojson",
    "customLidarTifPath": "/optional/path/to/custom/lidar.tif"
  }
}
```

### Required Fields
- `full_name` - Human-readable site name
- `epsg` - EPSG code for projection (e.g., 32613 for UTM Zone 13N)
- `inference_shape` - Filename of inference shapefile (in `downloads/{site}/infShp/`)
- `openTopoAPIkey` - OpenTopography API key for DEM downloads

### Optional Fields
- `notes` - Any notes about the site (training status, dates, etc.)
- `customTrainShpPath` - Path to custom training shapefile
- `customLidarTifPath` - Path to custom LiDAR data
- `maxarAPIkey` - Maxar API key (if different from OpenTopo key)

---

## Usage

### 1. With Command-Line `--site` Flag (Recommended)

When you use the `--site` flag, scripts automatically load configuration from `sites.json`:

```bash
# Data preparation
python prepTrainInputs/main1.py --site ws
python prepTrainInputs/main2.py --site ws

# Training
python ms_net/train.py --site ws

# Inference
python infer/main.py --site ws --scale mean
```

**No `.env` file copying needed!**

### 2. Programmatic Access

```python
from site_config import load_site_config

# Load site configuration
config = load_site_config('ws')

# Access configuration
epsg = config['epsg']
api_key = config['openTopoAPIkey']
inference_path = config['inferenceShpPath']
```

### 3. List Available Sites

```bash
python site_config.py
```

Output:
```
Available sites: ['ws', 'qm', 'lm']

============================================================
Site: ws - Wesner Springs
============================================================
EPSG: 32613
Inference Shape: wesner_springs_area_32613.geojson
Inference Path: /project/es14-stor/canopy/SatCHM/downloads/ws/infShp/wesner_springs_area_32613.geojson
Notes: Training completed 2026-07-28
============================================================
```

---

## Adding a New Site

### Step 1: Add Entry to sites.json

```json
{
  "new_site": {
    "full_name": "New Site Name",
    "epsg": 32613,
    "inference_shape": "new_site_area_32613.geojson",
    "openTopoAPIkey": "your_api_key_here",
    "notes": "Site added 2026-07-28"
  }
}
```

### Step 2: Create Inference Shapefile

Place your inference shapefile in:
```
downloads/new_site/infShp/new_site_area_32613.geojson
```

### Step 3: Run Workflow

```bash
python prepTrainInputs/main1.py --site new_site
python prepTrainInputs/main2.py --site new_site
python ms_net/train.py --site new_site
python infer/main.py --site new_site
```

---

## Migration from .env Files

### Current Workflow (Manual)
```bash
# Switch sites manually
cp .env_ws .env
python prepTrainInputs/main1.py

# Switch to different site
cp .env_qm .env
python prepTrainInputs/main1.py
```

### New Workflow (Automatic)
```bash
# No copying needed!
python prepTrainInputs/main1.py --site ws
python prepTrainInputs/main1.py --site qm
```

### Backward Compatibility

`.env` files still work as a fallback:
- If `--site` flag is provided → uses `sites.json`
- If no `--site` flag → falls back to `.env` file

---

## Security

### Protected Files
`sites.json` is added to `.gitignore` because it contains API keys.

### Template File
`sites.json.example` is provided as a template (tracked by git).

### Setup on New Machine
```bash
# Copy example file
cp sites.json.example sites.json

# Edit with your API keys
nano sites.json
```

---

## Troubleshooting

### Error: "Site 'xyz' not found in sites.json"
**Solution:** Add the site to `sites.json` or check spelling

### Error: "sites.json not found"
**Solution:** Create `sites.json` from `sites.json.example`

### Error: "Inference shape not found"
**Solution:** Verify the shapefile exists in `downloads/{site}/infShp/`

### Scripts still using .env
**Solution:** Make sure you're using the `--site` flag

---

## API Reference

### `load_site_config(site_code)`
Load configuration for a specific site.

**Parameters:**
- `site_code` (str): Site code (e.g., 'ws', 'qm', 'lm')

**Returns:**
- dict: Site configuration

**Example:**
```python
config = load_site_config('ws')
print(config['epsg'])  # 32613
```

### `get_site_env_vars(site_code)`
Get environment variables dict for backward compatibility.

**Parameters:**
- `site_code` (str): Site code

**Returns:**
- dict: Environment variable names to values

**Example:**
```python
env_vars = get_site_env_vars('ws')
print(env_vars['epsg'])  # '32613'
```

### `list_available_sites()`
List all available sites.

**Returns:**
- list: List of site codes

**Example:**
```python
sites = list_available_sites()
print(sites)  # ['ws', 'qm', 'lm']
```

### `print_site_info(site_code)`
Print detailed information about a site.

**Parameters:**
- `site_code` (str): Site code

**Example:**
```python
print_site_info('ws')
```

---

## Example sites.json

```json
{
  "ws": {
    "full_name": "Wesner Springs",
    "epsg": 32613,
    "inference_shape": "wesner_springs_area_32613.geojson",
    "openTopoAPIkey": "your_api_key_here",
    "notes": "Training completed 2026-07-28"
  },
  "qm": {
    "full_name": "Quemazon",
    "epsg": 32613,
    "inference_shape": "quemazon_area_32613.geojson",
    "openTopoAPIkey": "your_api_key_here",
    "notes": "Training completed 2026-07-24"
  },
  "lm": {
    "full_name": "Lookout Mountain",
    "epsg": 32613,
    "inference_shape": "lookout_mountain_area_32613.geojson",
    "openTopoAPIkey": "your_api_key_here",
    "notes": "Site configured, training pending"
  }
}
```

---

## Summary

**Old way:**
```bash
cp .env_ws .env && python main1.py
cp .env_qm .env && python main1.py
```

**New way:**
```bash
python main1.py --site ws
python main1.py --site qm
```

**Much simpler!** 🎉

---

*Last updated: 2026-07-28*
