"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare, derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

# chm-ms-net/infer/main1.py
from pathlib import Path
import sys
import os
import random
import shutil

from dotenv import load_dotenv

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_net.infer import run_inference


################### SETUP #################

load_dotenv()
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
epsg = int(os.getenv("epsg"))

# TODO: Change this to be extracted directly from lidar data
NORM_CONST = 46

################### CONFIG #################
NUM_TILES = 999999          # number of tiles to process per test site (None = all tiles)
RANDOM_SEED = 42

############### PATH DEFINITIONS ###############
weightsDir = "/mnt/c/Users/402630/Desktop/SatCHM_copy/weights_copy"
sites_data_dir = Path(project_path) / "sites_data"  # Master directory


################### DISCOVER ALL SITES ###################

def discover_sites(sites_data_dir):
    """
    Find all site subdirectories in sites_data/
    Returns dict: {site_name: site_path}
    """
    sites_path = Path(sites_data_dir)
    if not sites_path.exists():
        raise FileNotFoundError(f"Sites data directory not found: {sites_data_dir}")
    
    sites = {}
    for subdir in sites_path.iterdir():
        if subdir.is_dir() and subdir.name.endswith('_data'):
            site_name = subdir.name.replace('_data', '')
            # Check if it has required subdirectories
            if (subdir / 'chm').exists():
                sites[site_name] = subdir
            else:
                print(f"  ⚠️  Skipping {subdir.name}: missing 'chm' subdirectory")
    
    if not sites:
        raise FileNotFoundError(f"No valid site directories found in {sites_data_dir}")
    
    return sites


def get_weight_files(weights_dir):
    """
    Get all weight checkpoint files from directory.
    Looks for .ckpt, .pt, .pth files
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    
    weight_files = []
    for ext in ['*.ckpt', '*.pt', '*.pth']:
        weight_files.extend(weights_path.glob(ext))
    
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {weights_dir}")
    
    return sorted(weight_files)


def extract_tile_coords(filename):
    """
    Extract coordinates from filename like: 10300100632D9700_429296_3542800.tif
    Returns: "429296_3542800" or None if pattern doesn't match
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 3:
        coord1, coord2 = parts[-2], parts[-1]
        return f"{coord1}_{coord2}"
    return None


def organize_tile_by_coords(tile_path, base_output_dir):
    """
    Move tile to subdirectory named by its coordinates.
    Example: file_429296_3542800.tif -> 429296_3542800/file_429296_3542800.tif
    """
    tile_name = tile_path.name
    coords = extract_tile_coords(tile_name)
    
    if coords is None:
        print(f"  ⚠️  Could not extract coordinates from {tile_name}, leaving in root")
        return tile_path
    
    # Create coordinate-based subdirectory
    coord_dir = Path(base_output_dir) / coords
    coord_dir.mkdir(parents=True, exist_ok=True)
    
    # Move file to new location
    new_path = coord_dir / tile_name
    shutil.move(str(tile_path), str(new_path))
    
    return new_path


################ MAIN PROCESSING LOOP #################

# Discover all available sites
available_sites = discover_sites(sites_data_dir)
print(f"\n{'='*70}")
print(f"Discovered {len(available_sites)} test site(s):")
print(f"{'='*70}")
for site_name in sorted(available_sites.keys()):
    print(f"  • {site_name}")

# Get all weight files (trained models)
weight_files = get_weight_files(weightsDir)
print(f"\n{'='*70}")
print(f"Found {len(weight_files)} trained model(s):")
print(f"{'='*70}")
for wf in weight_files:
    print(f"  • {wf.name}")

print(f"\n{'='*70}")
print(f"STARTING CROSS-SITE INFERENCE")
print(f"Total combinations: {len(weight_files)} train sites × {len(available_sites)} test sites = {len(weight_files) * len(available_sites)}")
print(f"{'='*70}\n")

# Track results
results_summary = []

# For each trained model (train site)
for weight_idx, pathToWeights in enumerate(weight_files, 1):
    trainSite = Path(pathToWeights).stem
    print(f"\n{'='*70}")
    print(f"TRAIN SITE: {trainSite} [{weight_idx}/{len(weight_files)}]")
    print(f"{'='*70}\n")
    
    # Test on EACH available site
    for test_idx, (testSite, site_path) in enumerate(sorted(available_sites.items()), 1):
        print(f"{'─'*70}")
        print(f"Test Site {test_idx}/{len(available_sites)}: {testSite}")
        print(f"{'─'*70}")
        
        # Setup paths for this train-test combination
        inf_data_path = str(site_path)
        pred_tiles_dir = site_path / f"chm_preds_train_{trainSite}"  # Unique per train site
        true_tiles_dir = site_path / "chm"
        test_list_path = site_path / f"{testSite}_testlist.txt"
        
        # Create prediction directory
        pred_tiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {pred_tiles_dir}")
        
        # Handle test list subsetting if needed
        backup_exists = False
        original_tile_count = 0
        
        if test_list_path.exists():
            with open(test_list_path, "r") as f:
                all_tiles = [line.strip() for line in f if line.strip()]
            
            original_tile_count = len(all_tiles)
            
            if NUM_TILES is not None and len(all_tiles) > NUM_TILES:
                print(f"Limiting to {NUM_TILES} tiles (from {len(all_tiles)} available)")
                subset = sorted(random.Random(RANDOM_SEED).sample(all_tiles, NUM_TILES))
                
                temp_test_list = str(test_list_path) + ".subset"
                with open(temp_test_list, "w") as f:
                    f.write("\n".join(subset))
                
                test_list_path.rename(str(test_list_path) + ".original")
                Path(temp_test_list).rename(test_list_path)
                backup_exists = True
            else:
                print(f"Processing all {len(all_tiles)} tiles")
        else:
            print(f"⚠️  Warning: test list not found at {test_list_path}")
            print("    Inference may process all available tiles or fail")

        try:
            ################ RUN INFERENCE #######################
            print("\nRunning inference...")
            
            # Default location where run_inference saves files
            default_pred_dir = site_path / "chm_preds"
            
            run_inference(
                data_path=inf_data_path,
                site=testSite,
                NORM_CONST=NORM_CONST,
                model_loc=str(pathToWeights),
                epsg_code=epsg,
                phase="test",
                trainSite=trainSite,
                testSite=testSite,
            )
            print(f"✓ Inference complete")

            ################ MOVE FILES TO TRAIN-SPECIFIC DIRECTORY #################
            if default_pred_dir.exists() and default_pred_dir != pred_tiles_dir:
                print(f"\nMoving predictions from {default_pred_dir.name}/ to {pred_tiles_dir.name}/...")
                
                # Get all tif files from default location
                tif_files = list(default_pred_dir.glob("*.tif"))
                
                if tif_files:
                    # Ensure target directory exists
                    pred_tiles_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move each file
                    for tif_file in tif_files:
                        dest = pred_tiles_dir / tif_file.name
                        shutil.move(str(tif_file), str(dest))
                    
                    print(f"✓ Moved {len(tif_files)} files")
                    
                    # Clean up empty default directory if it's not needed
                    try:
                        if not any(default_pred_dir.iterdir()):
                            default_pred_dir.rmdir()
                            print(f"✓ Cleaned up empty {default_pred_dir.name}/ directory")
                    except:
                        pass
                else:
                    print("  ⚠️  No .tif files found in default prediction directory")
            
            # Count final tiles
            final_tiles = list(pred_tiles_dir.glob("*.tif"))
            print(f"✓ Total tiles generated: {len(final_tiles)}")
            
            results_summary.append({
                'train_site': trainSite,
                'test_site': testSite,
                'tiles_generated': len(final_tiles),
                'output_dir': str(pred_tiles_dir),
                'status': 'success'
            })

        except Exception as e:
            print(f"\n❌ Error processing {trainSite} → {testSite}:")
            print(f"   {str(e)}")
            results_summary.append({
                'train_site': trainSite,
                'test_site': testSite,
                'tiles_generated': 0,
                'output_dir': str(pred_tiles_dir),
                'status': f'failed: {str(e)}'
            })
            
        finally:
            # Restore original test list
            if backup_exists:
                if test_list_path.exists():
                    test_list_path.unlink()
                Path(str(test_list_path) + ".original").rename(test_list_path)
                print("✓ Restored original test list")
        
        print()  # Blank line between test sites


################ FINAL SUMMARY #################

print("\n" + "="*70)
print("="*70)
print("CROSS-SITE INFERENCE COMPLETE")
print("="*70)
print("="*70 + "\n")

# Create summary table
test_sites = sorted(available_sites.keys())
train_sites = sorted([Path(w).stem for w in weight_files])

print("Generation Summary (tiles created):\n")
header = f"{'Train \\ Test':<15}"
for ts in test_sites:
    header += f"{ts:<12}"
print(header)
print("-" * (15 + 12 * len(test_sites)))

for train_site in train_sites:
    row = f"{train_site:<15}"
    for test_site in test_sites:
        # Find result for this combination
        result = next((r for r in results_summary 
                      if r['train_site'] == train_site and r['test_site'] == test_site), None)
        if result and result['status'] == 'success':
            row += f"{result['tiles_generated']:>10}  "
        else:
            row += f"{'FAIL':<12}"
    print(row)

print("\n" + "─"*70 + "\n")

# Total statistics
successful = [r for r in results_summary if r['status'] == 'success']
failed = [r for r in results_summary if r['status'] != 'success']
total_tiles = sum(r['tiles_generated'] for r in successful)

print(f"Total combinations processed: {len(results_summary)}")
print(f"  ✓ Successful: {len(successful)}")
print(f"  ✗ Failed: {len(failed)}")
print(f"\nTotal prediction tiles generated: {total_tiles:,}")

if failed:
    print(f"\n⚠️  Failed combinations:")
    for r in failed:
        print(f"  • {r['train_site']} → {r['test_site']}: {r['status']}")

print(f"\n{'='*70}")
print("All predictions saved to respective sites_data/*/chm_preds_train_* directories")
print("="*70)