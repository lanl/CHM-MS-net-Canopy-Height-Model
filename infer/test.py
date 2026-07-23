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
from dotenv import load_dotenv

# Add the project root (the folder that contains `prepTrainInputs/`) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import SatCHM.prepTrainInputs.utils as utils
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
NUM_TILES = 10          # number of tiles to process end-to-end
RANDOM_SEED = 42
MAX_MATCH_DIST = None   # set to a numeric value in CRS units if you want to reject distant matches

############### PATH DEFINITIONS ###############
inf_data_path = os.path.join(project_path, f"{site}_data")
pred_tiles_dir = os.path.join(inf_data_path, "chm_preds")
true_tiles_dir = os.path.join(inf_data_path, "chm")
test_list_path = os.path.join(inf_data_path, f"{site}_testlist.txt")

pathToWeights = "/mnt/c/Users/402630/Desktop/SatCHM_copy/jointBase.ckpt"  # fairbanks weights
rdsPath = None  # Optional, only needed if trying to compute CBH in treelist

# folder creations
os.makedirs(inf_data_path, exist_ok=True)


################### HELPERS ###################

def choose_matching_tiles(pred_dir, true_dir, test_list_file=None, sample_size=10, seed=42):
    """
    Select up to `sample_size` tile filenames that exist in both pred_dir and true_dir.
    If test_list_file exists, restrict selection to filenames in that list.
    """
    pred_files = {p.name for p in Path(pred_dir).glob("*.tif")}
    true_files = {p.name for p in Path(true_dir).glob("*.tif")}

    common = pred_files & true_files

    if test_list_file and os.path.exists(test_list_file):
        with open(test_list_file, "r") as f:
            allowed = {line.strip() for line in f if line.strip()}
        common &= allowed

    common = sorted(common)

    if len(common) == 0:
        raise RuntimeError("No matching tile filenames found between chm_preds and chm.")

    rng = random.Random(seed)
    if len(common) <= sample_size:
        return common

    return sorted(rng.sample(common, sample_size))


def run_treelist_for_single_tile(tif_path, out_csv_path, project_path, epsg, rdsPath=None):
    """
    Run utils.genTreelist on a single tile and ensure the produced CSV ends up at out_csv_path.
    """
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    filename_only = out_csv_path.name

    if rdsPath is not None:
        utils.genTreelist(
            tifPath=str(tif_path),
            projectPath=project_path,
            rdsPath=rdsPath,
            epsg=epsg,
            filename=filename_only,
        )
    else:
        print(f"filename_only: {filename_only}")
        utils.genTreelist(
            tifPath=str(tif_path),
            projectPath=project_path,
            epsg=epsg,
            filename=filename_only,
        )

    candidate_paths = [
        out_csv_path,
        Path(project_path) / filename_only,
        Path(tif_path).parent / filename_only,
        Path(inf_data_path) / filename_only,
    ]

    existing = [p for p in candidate_paths if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"genTreelist finished but could not find output CSV for {tif_path}. "
            f"Looked for: {[str(p) for p in candidate_paths]}"
        )

    src = existing[0]
    if src.resolve() != out_csv_path.resolve():
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


################ RUN INFERENCE #######################

print("Running inference")
run_inference(
    data_path=inf_data_path,
    site=site,
    NORM_CONST=NORM_CONST,
    model_loc=pathToWeights,
    epsg_code=epsg,
    phase="test",
)
print(f"Ran inference, tiles saved to {pred_tiles_dir}")


############# TILE-BY-TILE TREE MATCHING ####################

print(f"Selecting up to {NUM_TILES} matching tiles from chm_preds and chm...")
selected_tiles = choose_matching_tiles(
    pred_dir=pred_tiles_dir,
    true_dir=true_tiles_dir,
    test_list_file=test_list_path,
    sample_size=NUM_TILES,
    seed=RANDOM_SEED,
)

print(f"Selected {len(selected_tiles)} tiles:")
for tile_name in selected_tiles:
    print(f"  {tile_name}")

treelist_tmp_dir = Path(tempfile.mkdtemp(prefix="tile_treelists_"))

all_height_diffs = []
all_area_diffs = []
all_match_dists = []
all_pred_heights = []
all_true_heights = []

for i, tile_name in enumerate(selected_tiles, start=1):
    pred_tile = Path(pred_tiles_dir) / tile_name
    true_tile = Path(true_tiles_dir) / tile_name

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
    print("\nNo matched trees were found across the selected tiles.")
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

    print("\n================ FINAL RESULTS ================")
    print(f"Tiles processed: {len(selected_tiles)}")
    print(f"Matched trees total: {len(all_height_diffs)}")

    print("\n--- Height Metrics ---")
    print(f"Mean signed height diff (pred - true): {overall_mean_signed:.3f}")
    print(f"Mean absolute height diff: {overall_mean_abs:.3f}")
    print(f"Std dev of height differences: {overall_std:.3f}")

    print("\n--- Crown Area Metrics ---")
    if mean_abs_area_diff is not None:
        print(f"Mean absolute crown area diff: {mean_abs_area_diff:.3f}")
    else:
        print("Mean absolute crown area diff: not available")

    print("\n--- Tree Height Stats ---")
    print(f"Mean predicted tree height: {mean_pred_height:.3f}")
    print(f"Mean true tree height: {mean_true_height:.3f}")

    print("\n--- Matching Quality ---")
    print(f"Mean centroid match distance: {overall_mean_dist:.3f}")

    print("==============================================")