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
import tempfile

import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv

# Add the project root (the folder that contains `prepTrainInputs/`) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepTrainInputs.main1Utils as utils
from ms_net.infer import run_inference


################### SETUP #################

load_dotenv()
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
chmPath = os.getenv("chmPath")
chmReducedPath = os.getenv("chmReducedPath")
shpPath = os.getenv("shpPath")
epsg = int(os.getenv("epsg"))
site = os.getenv("site")
inferenceShpPath = os.getenv("inferenceShpPath")
customTrainShpPath = os.getenv("customTrainShpPath")
fp_path = os.getenv("fp_path")
openTopoAPIkey = os.getenv("openTopoAPIkey")
maxarAPIkey = os.getenv("maxarAPIkey")
customLidarTifPath = os.getenv("customLidarTifPath")


# TODO: Change this to be extracted directly from lidar data
NORM_CONST = 46

################### CONFIG #################
NUM_TILES = 5          # number of tiles to process per weight file
RANDOM_SEED = 42
MAX_MATCH_DIST = None   # set to a numeric value in CRS units if you want to reject distant matches
USE_PIXELWISE_METRICS = True  # If True, skip tree segmentation and compute pixelwise stats

############### PATH DEFINITIONS ###############
# Changed from single weight file to directory
weightsDir = "/mnt/c/Users/402630/Desktop/SatCHM_copy/weights_copy"

inf_data_path = os.path.join(project_path, f"{site}_data")
pred_tiles_dir = os.path.join(inf_data_path, "chm_preds")
true_tiles_dir = os.path.join(inf_data_path, "chm")
test_list_path = os.path.join(inf_data_path, f"{site}_testlist.txt")

rdsPath = None  # Optional, only needed if trying to compute CBH in treelist

# folder creations
os.makedirs(inf_data_path, exist_ok=True)

testSite = site


################### HELPERS ###################

def extract_tile_coords(filename):
    """
    Extract coordinates from filename like: 10300100632D9700_429296_3542800.tif
    Returns: (429296, 3542800) as strings or None if pattern doesn't match
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 3:
        # Get last two parts
        coord1, coord2 = parts[-2], parts[-1]
        return f"{coord1}_{coord2}"
    return None


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


def organize_tile_by_coords(tile_path, base_output_dir):
    """
    Move tile to subdirectory named by its coordinates.
    Example: file_429296_3542800.tif -> chm_preds/429296_3542800/file_429296_3542800.tif
    """
    tile_name = tile_path.name
    coords = extract_tile_coords(tile_name)
    
    if coords is None:
        print(f"  ⚠️  Could not extract coordinates from {tile_name}, skipping organization")
        return tile_path
    
    # Create coordinate-based subdirectory
    coord_dir = Path(base_output_dir) / coords
    coord_dir.mkdir(parents=True, exist_ok=True)
    
    # Move file to new location
    new_path = coord_dir / tile_name
    shutil.move(str(tile_path), str(new_path))
    
    return new_path


def get_processed_tiles_with_validation(pred_dir, true_dir):
    """
    Get tiles that were processed AND have ground truth data available.
    """
    pred_files = {p.name for p in Path(pred_dir).rglob("*.tif")}
    
    valid_tiles = []
    skipped = []
    
    for tile_name in sorted(pred_files):
        # Check if corresponding ground truth exists
        true_matches = list(Path(true_dir).rglob(tile_name))
        if true_matches:
            valid_tiles.append(tile_name)
        else:
            skipped.append(tile_name)
    
    if skipped:
        print(f"  ⚠️  Skipping {len(skipped)} tile(s) without ground truth")
    
    if not valid_tiles:
        raise RuntimeError("No tiles found with matching ground truth")
    
    return valid_tiles


def find_tile_path(base_dir, tile_name):
    """
    Find full path to a tile that might be in a subdirectory.
    """
    matches = list(Path(base_dir).rglob(tile_name))
    if not matches:
        raise FileNotFoundError(f"Could not find {tile_name} in {base_dir}")
    return matches[0]


def compute_pixelwise_metrics(pred_tif, true_tif):
    """
    Compute pixelwise difference statistics between two rasters.
    Returns dict with mean, std, percentiles, etc.
    """
    with rasterio.open(pred_tif) as pred_src:
        pred_data = pred_src.read(1).astype(float)
        pred_nodata = pred_src.nodata
    
    with rasterio.open(true_tif) as true_src:
        true_data = true_src.read(1).astype(float)
        true_nodata = true_src.nodata
    
    # Mask out nodata values
    valid_mask = np.ones_like(pred_data, dtype=bool)
    if pred_nodata is not None:
        valid_mask &= (pred_data != pred_nodata)
    if true_nodata is not None:
        valid_mask &= (true_data != true_nodata)
    
    pred_valid = pred_data[valid_mask]
    true_valid = true_data[valid_mask]
    
    if len(pred_valid) == 0:
        return None
    
    diff = pred_valid - true_valid
    
    # Filter out NaN and Inf values from the difference array
    finite_mask = np.isfinite(diff)
    diff_finite = diff[finite_mask]
    
    if len(diff_finite) == 0:
        return None
    
    return {
        'n_pixels': len(diff_finite),
        'mean_diff': float(np.mean(diff_finite)),
        'mean_abs_diff': float(np.mean(np.abs(diff_finite))),
        'std_diff': float(np.std(diff_finite)),
        'median_diff': float(np.median(diff_finite)),
        'p5': float(np.percentile(diff_finite, 5)),
        'p95': float(np.percentile(diff_finite, 95)),
        'min_diff': float(np.min(diff_finite)),
        'max_diff': float(np.max(diff_finite)),
    }


def run_treelist_for_single_tile(tif_path, out_csv_path, project_path, epsg, rdsPath=None):
    """
    Run utils.genTreelist on a single tile and ensure the produced CSV ends up at out_csv_path.
    """
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Let genTreelist create the file with its default name first
    if rdsPath is not None:
        utils.genTreelist(
            tifPath=str(tif_path),
            projectPath=project_path,
            rdsPath=rdsPath,
            epsg=epsg,
        )
    else:
        utils.genTreelist(
            tifPath=str(tif_path),
            projectPath=project_path,
            epsg=epsg,
        )

    # Now search for the generated file in common locations
    candidate_paths = [
        Path(tif_path).parent / "treelist.csv",  # Same dir as tif
        Path(project_path) / "treelist.csv",     # Project root
        Path(inf_data_path) / "treelist.csv",    # Data dir
    ]

    existing = [p for p in candidate_paths if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"genTreelist finished but could not find treelist.csv for {tif_path}. "
            f"Looked for: {[str(p) for p in candidate_paths]}"
        )

    src = existing[0]
    shutil.move(str(src), str(out_csv_path))
    
    return out_csv_path


def find_column(df, candidates, required=True):
    """
    Return the first matching column from candidates, case-insensitive.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find any of columns {candidates} in treelist. "
            f"Found columns: {list(df.columns)}"
        )
    return None


def load_treelist_csv(csv_path):
    """
    Load treelist CSV and normalize to columns: x, y, height, crown_area
    """
    df = pd.read_csv(csv_path)

    x_col = find_column(df, ["x", "utm_x", "xcoord", "x_coord", "centroid_x", "tree_x"])
    y_col = find_column(df, ["y", "utm_y", "ycoord", "y_coord", "centroid_y", "tree_y"])
    h_col = find_column(df, ["height", "ht", "tree_height", "h", "tree_height_m"])
    area_col = find_column(
        df,
        ["crown_area_m2", "crown_area", "area", "crownarea", "crownArea"],
        required=False
    )

    cols = [x_col, y_col, h_col]
    names = ["x", "y", "height"]

    if area_col is not None:
        cols.append(area_col)
        names.append("crown_area")

    out = df[cols].copy()
    out.columns = names
    out = out.dropna(subset=["x", "y", "height"]).reset_index(drop=True)
    return out


def nearest_neighbor_metrics(pred_df, true_df, max_dist=None):
    """
    For each predicted tree centroid, find nearest true-tree centroid.

    Returns:
      height_diffs: pred_height - true_height
      area_diffs: pred_crown_area - true_crown_area, or None if not available
      dists: nearest-neighbor centroid distances
    """
    if pred_df.empty or true_df.empty:
        return np.array([]), None, np.array([])

    pred_xy = pred_df[["x", "y"]].to_numpy(dtype=float)
    true_xy = true_df[["x", "y"]].to_numpy(dtype=float)

    dists = np.sqrt(((pred_xy[:, None, :] - true_xy[None, :, :]) ** 2).sum(axis=2))
    nn_idx = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(len(pred_df)), nn_idx]

    pred_h = pred_df["height"].to_numpy(dtype=float)
    true_h = true_df.iloc[nn_idx]["height"].to_numpy(dtype=float)
    height_diffs = pred_h - true_h

    area_diffs = None
    if "crown_area" in pred_df.columns and "crown_area" in true_df.columns:
        pred_a = pred_df["crown_area"].to_numpy(dtype=float)
        true_a = true_df.iloc[nn_idx]["crown_area"].to_numpy(dtype=float)
        area_diffs = pred_a - true_a

    if max_dist is not None:
        keep = nn_dist <= max_dist
        height_diffs = height_diffs[keep]
        nn_dist = nn_dist[keep]
        if area_diffs is not None:
            area_diffs = area_diffs[keep]

    return height_diffs, area_diffs, nn_dist


################ MAIN PROCESSING LOOP #################

# Get all weight files
weight_files = get_weight_files(weightsDir)
print(f"Found {len(weight_files)} weight file(s) to process:")
for wf in weight_files:
    print(f"  - {wf.name}")

# Store results for each site
all_site_results = {}

for weight_idx, pathToWeights in enumerate(weight_files, 1):
    trainSite = Path(pathToWeights).stem
    print(f"\n{'='*70}")
    print(f"Processing weight file {weight_idx}/{len(weight_files)}: {trainSite}")
    print(f"{'='*70}\n")
    
    ################ APPLY DATA CAP BEFORE INFERENCE #################
    backup_exists = False
    
    # Create subset test list file before inference
    if os.path.exists(test_list_path):
        with open(test_list_path, "r") as f:
            all_tiles = [line.strip() for line in f if line.strip()]
        
        if len(all_tiles) > NUM_TILES:
            print(f"Capping data: selecting {NUM_TILES} from {len(all_tiles)} tiles")
            subset = sorted(random.Random(RANDOM_SEED).sample(all_tiles, NUM_TILES))
            
            temp_test_list = test_list_path + ".subset"
            with open(temp_test_list, "w") as f:
                f.write("\n".join(subset))
            
            os.rename(test_list_path, test_list_path + ".original")
            os.rename(temp_test_list, test_list_path)
            backup_exists = True

    try:
        ################ RUN INFERENCE #######################
        print("Running inference...")
        run_inference(
            data_path=inf_data_path,
            site=site,
            NORM_CONST=NORM_CONST,
            model_loc=str(pathToWeights),
            epsg_code=epsg,
            phase="test",
            trainSite=trainSite,
            testSite=testSite,
        )
        print(f"Inference complete. Raw tiles saved to {pred_tiles_dir}")

        ################ ORGANIZE TILES BY COORDINATES #################
        print("\nOrganizing tiles by coordinates...")
        pred_tiles = list(Path(pred_tiles_dir).glob("*.tif"))
        
        organized_count = 0
        for tile_path in pred_tiles:
            coords = extract_tile_coords(tile_path.name)
            if coords:
                organize_tile_by_coords(tile_path, pred_tiles_dir)
                organized_count += 1
        
        print(f"Organized {organized_count} tiles into coordinate-based subdirectories")

        ############# GET PROCESSED TILES ####################
        print(f"\nGetting processed tiles from inference...")
        selected_tiles = get_processed_tiles_with_validation(pred_tiles_dir, true_tiles_dir)

        print(f"Found {len(selected_tiles)} tiles to evaluate:")
        for tile_name in selected_tiles:
            coords = extract_tile_coords(tile_name)
            print(f"  {tile_name} → {coords}/")

        print(f"Selected {len(selected_tiles)} tiles:")
        for tile_name in selected_tiles:
            coords = extract_tile_coords(tile_name)
            print(f"  {tile_name} → {coords}/")

        ############# EVALUATION MODE SELECTION ####################
        
        if USE_PIXELWISE_METRICS:
            # ===== PIXELWISE MODE =====
            print(f"\n=== PIXELWISE EVALUATION MODE (Site: {trainSite}) ===\n")
            
            all_tile_stats = []
            
            for i, tile_name in enumerate(selected_tiles, start=1):
                pred_tile = find_tile_path(pred_tiles_dir, tile_name)
                true_tile = find_tile_path(true_tiles_dir, tile_name)
                
                print(f"[{i}/{len(selected_tiles)}] Processing: {tile_name}")
                
                stats = compute_pixelwise_metrics(pred_tile, true_tile)
                
                if stats is None:
                    print("  ⚠️  No valid pixels found")
                    continue
                
                all_tile_stats.append(stats)
                
                print(f"  Pixels: {stats['n_pixels']:,}")
                print(f"  Mean diff: {stats['mean_diff']:+.3f} (MAE: {stats['mean_abs_diff']:.3f})")
                print(f"  Std dev: {stats['std_diff']:.3f}")
                print(f"  Median: {stats['median_diff']:+.3f}")
                print(f"  Range: [{stats['min_diff']:+.3f}, {stats['max_diff']:+.3f}]")
                print(f"  5th-95th percentile: [{stats['p5']:+.3f}, {stats['p95']:+.3f}]\n")
            
            if len(all_tile_stats) == 0:
                print(f"\nNo valid tile statistics computed for {trainSite}.")
                all_site_results[trainSite] = None
            else:
                # Aggregate across all tiles
                total_pixels = sum(s['n_pixels'] for s in all_tile_stats)
                weighted_mean = sum(s['mean_diff'] * s['n_pixels'] for s in all_tile_stats) / total_pixels
                weighted_mean_abs = sum(s['mean_abs_diff'] * s['n_pixels'] for s in all_tile_stats) / total_pixels
                
                site_results = {
                    'tiles_processed': len(all_tile_stats),
                    'total_pixels': total_pixels,
                    'weighted_mean_diff': weighted_mean,
                    'weighted_mean_abs': weighted_mean_abs,
                    'mean_of_tile_means': np.mean([s['mean_diff'] for s in all_tile_stats]),
                    'mean_of_tile_stds': np.mean([s['std_diff'] for s in all_tile_stats]),
                    'median_of_tile_medians': np.median([s['median_diff'] for s in all_tile_stats]),
                }
                
                all_site_results[trainSite] = site_results
                
                print(f"\n================ PIXELWISE RESULTS ({trainSite}) ================")
                print(f"Tiles processed: {site_results['tiles_processed']}")
                print(f"Total pixels evaluated: {site_results['total_pixels']:,}")
                print(f"\nWeighted mean diff (pred - true): {site_results['weighted_mean_diff']:+.3f}")
                print(f"Weighted mean absolute error: {site_results['weighted_mean_abs']:.3f}")
                print(f"\nPer-tile statistics:")
                print(f"  Mean of tile means: {site_results['mean_of_tile_means']:+.3f}")
                print(f"  Mean of tile std devs: {site_results['mean_of_tile_stds']:.3f}")
                print(f"  Median of tile medians: {site_results['median_of_tile_medians']:+.3f}")
                print("================================================================\n")
        
        else:
                        # ===== TREE-BASED MODE =====
            print(f"\n=== TREE-BASED EVALUATION MODE (Site: {trainSite}) ===\n")
            
            treelist_tmp_dir = Path(tempfile.mkdtemp(prefix=f"tile_treelists_{trainSite}_"))

            all_height_diffs = []
            all_area_diffs = []
            all_match_dists = []
            all_pred_heights = []
            all_true_heights = []

            for i, tile_name in enumerate(selected_tiles, start=1):
                pred_tile = find_tile_path(pred_tiles_dir, tile_name)
                true_tile = find_tile_path(true_tiles_dir, tile_name)

                pred_csv = treelist_tmp_dir / f"pred_{i:02d}_{Path(tile_name).stem}_treelist.csv"
                true_csv = treelist_tmp_dir / f"true_{i:02d}_{Path(tile_name).stem}_treelist.csv"

                print(f"\n[{i}/{len(selected_tiles)}] Processing tile: {tile_name}")

                print("  Generating predicted treelist...")
                run_treelist_for_single_tile(
                    tif_path=pred_tile,
                    out_csv_path=pred_csv,
                    project_path=project_path,
                    epsg=epsg,
                    rdsPath=rdsPath,
                )

                print("  Generating true treelist...")
                run_treelist_for_single_tile(
                    tif_path=true_tile,
                    out_csv_path=true_csv,
                    project_path=project_path,
                    epsg=epsg,
                    rdsPath=rdsPath,
                )

                pred_df = load_treelist_csv(pred_csv)
                true_df = load_treelist_csv(true_csv)

                print(f"  Pred trees: {len(pred_df)}, True trees: {len(true_df)}")

                height_diffs, area_diffs, match_dists = nearest_neighbor_metrics(
                    pred_df=pred_df,
                    true_df=true_df,
                    max_dist=MAX_MATCH_DIST,
                )

                if len(height_diffs) == 0:
                    print("  No matches found for this tile.")
                    continue

                tile_mean_signed = float(np.mean(height_diffs))
                tile_mean_abs = float(np.mean(np.abs(height_diffs)))
                tile_std = float(np.std(height_diffs))
                tile_mean_dist = float(np.mean(match_dists)) if len(match_dists) > 0 else float("nan")

                print(f"  Tile matched trees: {len(height_diffs)}")
                print(f"  Tile mean signed height diff (pred - true): {tile_mean_signed:.3f}")
                print(f"  Tile mean absolute height diff: {tile_mean_abs:.3f}")
                print(f"  Tile std dev of height differences: {tile_std:.3f}")

                if area_diffs is not None and len(area_diffs) > 0:
                    tile_mean_abs_area = float(np.mean(np.abs(area_diffs)))
                    print(f"  Tile mean absolute crown area diff: {tile_mean_abs_area:.3f}")
                else:
                    print("  Tile mean absolute crown area diff: not available")

                print(f"  Tile mean centroid match distance: {tile_mean_dist:.3f}")

                all_height_diffs.extend(height_diffs.tolist())
                all_match_dists.extend(match_dists.tolist())

                if area_diffs is not None:
                    all_area_diffs.extend(area_diffs.tolist())

                all_pred_heights.extend(pred_df["height"].tolist())
                all_true_heights.extend(true_df["height"].tolist())

            if len(all_height_diffs) == 0:
                print(f"\nNo matched trees were found across the selected tiles for {trainSite}.")
                all_site_results[trainSite] = None
            else:
                all_height_diffs = np.array(all_height_diffs, dtype=float)
                all_match_dists = np.array(all_match_dists, dtype=float)

                overall_mean_signed = float(np.mean(all_height_diffs))
                overall_mean_abs = float(np.mean(np.abs(all_height_diffs)))
                overall_std = float(np.std(all_height_diffs))
                overall_mean_dist = float(np.mean(all_match_dists)) if len(all_match_dists) > 0 else float("nan")

                if len(all_area_diffs) > 0:
                    all_area_diffs = np.array(all_area_diffs, dtype=float)
                    mean_abs_area_diff = float(np.mean(np.abs(all_area_diffs)))
                else:
                    mean_abs_area_diff = None

                mean_pred_height = float(np.mean(all_pred_heights)) if len(all_pred_heights) > 0 else float("nan")
                mean_true_height = float(np.mean(all_true_heights)) if len(all_true_heights) > 0 else float("nan")

                site_results = {
                    'tiles_processed': len(selected_tiles),
                    'matched_trees': len(all_height_diffs),
                    'mean_signed_height_diff': overall_mean_signed,
                    'mean_abs_height_diff': overall_mean_abs,
                    'std_height_diff': overall_std,
                    'mean_abs_area_diff': mean_abs_area_diff,
                    'mean_pred_height': mean_pred_height,
                    'mean_true_height': mean_true_height,
                    'mean_match_dist': overall_mean_dist,
                }
                
                all_site_results[trainSite] = site_results

                print(f"\n================ TREE-BASED RESULTS ({trainSite}) ================")
                print(f"Tiles processed: {site_results['tiles_processed']}")
                print(f"Matched trees total: {site_results['matched_trees']}")

                print("\n--- Height Metrics ---")
                print(f"Mean signed height diff (pred - true): {site_results['mean_signed_height_diff']:.3f}")
                print(f"Mean absolute height diff: {site_results['mean_abs_height_diff']:.3f}")
                print(f"Std dev of height differences: {site_results['std_height_diff']:.3f}")

                print("\n--- Crown Area Metrics ---")
                if site_results['mean_abs_area_diff'] is not None:
                    print(f"Mean absolute crown area diff: {site_results['mean_abs_area_diff']:.3f}")
                else:
                    print("Mean absolute crown area diff: not available")

                print("\n--- Tree Height Stats ---")
                print(f"Mean predicted tree height: {site_results['mean_pred_height']:.3f}")
                print(f"Mean true tree height: {site_results['mean_true_height']:.3f}")

                print("\n--- Matching Quality ---")
                print(f"Mean centroid match distance: {site_results['mean_match_dist']:.3f}")

                print("===================================================================\n")

    finally:
        # Restore original test list
        if backup_exists:
            if os.path.exists(test_list_path):
                os.remove(test_list_path)
            os.rename(test_list_path + ".original", test_list_path)
            print(f"Restored original test list for {trainSite}")


################ SUMMARY OF ALL SITES #################

print("\n" + "="*70)
print("="*70)
print("SUMMARY: ALL SITES")
print("="*70)
print("="*70 + "\n")

if not all_site_results:
    print("No results to summarize.")
else:
    valid_sites = {k: v for k, v in all_site_results.items() if v is not None}
    
    if not valid_sites:
        print("All sites failed to produce valid results.")
    else:
        print(f"Successfully processed {len(valid_sites)}/{len(all_site_results)} site(s)\n")
        
        if USE_PIXELWISE_METRICS:
            print("┌─────────────────────────────────────────────────────────────────┐")
            print("│                    PIXELWISE COMPARISON                         │")
            print("└─────────────────────────────────────────────────────────────────┘\n")
            
            # Create comparison table
            print(f"{'Site':<20} {'Tiles':<8} {'Pixels':<12} {'Mean Diff':<12} {'MAE':<12}")
            print("-" * 70)
            
            for site_name, results in valid_sites.items():
                print(f"{site_name:<20} {results['tiles_processed']:<8} "
                      f"{results['total_pixels']:<12,} "
                      f"{results['weighted_mean_diff']:>+11.3f} "
                      f"{results['weighted_mean_abs']:>11.3f}")
            
            print("\n" + "-" * 70)
            
            # Best performing site
            best_site = min(valid_sites.items(), 
                          key=lambda x: x[1]['weighted_mean_abs'])
            
            print(f"\n🏆 Best performing site (lowest MAE): {best_site[0]}")
            print(f"   MAE: {best_site[1]['weighted_mean_abs']:.3f}")
            
            # Average across all sites
            avg_mae = np.mean([r['weighted_mean_abs'] for r in valid_sites.values()])
            avg_mean = np.mean([r['weighted_mean_diff'] for r in valid_sites.values()])
            
            print(f"\n📊 Average across all sites:")
            print(f"   Mean diff: {avg_mean:+.3f}")
            print(f"   MAE: {avg_mae:.3f}")
            
        else:
            print("┌─────────────────────────────────────────────────────────────────┐")
            print("│                     TREE-BASED COMPARISON                       │")
            print("└─────────────────────────────────────────────────────────────────┘\n")
            
            # Create comparison table
            print(f"{'Site':<20} {'Tiles':<8} {'Trees':<10} {'Mean Ht Diff':<14} {'MAE Ht':<12}")
            print("-" * 70)
            
            for site_name, results in valid_sites.items():
                print(f"{site_name:<20} {results['tiles_processed']:<8} "
                      f"{results['matched_trees']:<10} "
                      f"{results['mean_signed_height_diff']:>+13.3f} "
                      f"{results['mean_abs_height_diff']:>11.3f}")
            
            print("\n" + "-" * 70)
            
            # Best performing site
            best_site = min(valid_sites.items(), 
                          key=lambda x: x[1]['mean_abs_height_diff'])
            
            print(f"\n🏆 Best performing site (lowest height MAE): {best_site[0]}")
            print(f"   Height MAE: {best_site[1]['mean_abs_height_diff']:.3f}")
            print(f"   Matched trees: {best_site[1]['matched_trees']}")
            
            # Average across all sites
            avg_height_mae = np.mean([r['mean_abs_height_diff'] for r in valid_sites.values()])
            avg_height_mean = np.mean([r['mean_signed_height_diff'] for r in valid_sites.values()])
            total_trees = sum([r['matched_trees'] for r in valid_sites.values()])
            
            print(f"\n📊 Average across all sites:")
            print(f"   Mean height diff: {avg_height_mean:+.3f}")
            print(f"   Height MAE: {avg_height_mae:.3f}")
            print(f"   Total trees matched: {total_trees:,}")

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print(f"\nPredictions organized in: {pred_tiles_dir}")
print("Each tile is stored in a subdirectory named by its coordinates.")