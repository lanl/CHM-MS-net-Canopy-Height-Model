# © 2026. Triad National Security, LLC. All rights reserved.
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.


#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(stringr)
  library(lidR)
  library(terra)
  library(lutz)
  library(sf)
  library(jsonlite)
  library(docopt)
  library(randomForest)
  # optional to attach; the code uses namespaced calls:
  # library(readr)
  # library(dplyr)
})

# ---------- utils ----------
chm_create <- function(filename, epsg, output_dir = "data") {
  nums <- as.numeric(unlist(stringr::str_extract_all(basename(filename), "\\d+")))
  if (length(nums) < 2) return(paste(filename, "error filename-parse"))
  x_ul <- nums[1]; y_ul <- nums[2]

  tile_size <- 256
  res_m     <- 0.5
  buf       <- 20
  utm_crs   <- paste0("EPSG:", as.integer(epsg))

  lidR::set_lidr_threads(2)
  las <- try(lidR::readLAS(filename, filter = "-drop_withheld -drop_class 7 18"), silent = TRUE)
  if (inherits(las, "try-error") || lidR::is.empty(las)) return(paste(filename, "error empty-las"))

  las <- lidR::classify_noise(las, lidR::ivf(res = 5, n = 6))
  las <- lidR::filter_poi(las, Classification != 18)
  if (lidR::is.empty(las)) return(paste(filename, "error empty-noise"))

  nlas <- try(lidR::normalize_height(las, lidR::knnidw()), silent = TRUE)
  if (inherits(nlas, "try-error") || lidR::is.empty(nlas)) return(paste(filename, "error normalize"))
  nlas <- lidR::filter_poi(nlas, Z >= 0 & Z < 300)
  if (lidR::is.empty(nlas)) return(paste(filename, "error empty-filter"))

  ext_tile <- terra::ext(x_ul, x_ul + tile_size, y_ul - tile_size, y_ul)
  ext_buf  <- terra::ext(x_ul - buf, x_ul + tile_size + buf, y_ul - tile_size - buf, y_ul + buf)

  tmpl_buf  <- terra::rast(extent = ext_buf,  resolution = res_m, crs = utm_crs)
  tmpl_tile <- terra::rast(extent = ext_tile, resolution = res_m, crs = utm_crs)

  chm_buf <- try(
    lidR::rasterize_canopy(
      las = nlas,
      algorithm = lidR::pitfree(thresholds = c(0, 2, 5, 10, 15), max_edge = c(10, 1), subcircle = 0.35),
      raster = tmpl_buf, pkg = "terra"
    ), silent = TRUE
  )
  if (inherits(chm_buf, "try-error")) {
    chm_buf <- try(lidR::rasterize_canopy(nlas, lidR::p2r(subcircle = 0.35), raster = tmpl_buf, pkg = "terra"), silent = TRUE)
  }
  if (inherits(chm_buf, "try-error")) return(paste(filename, "error chm"))

  chm <- terra::resample(chm_buf, tmpl_tile, method = "bilinear")

  gps_time <- try(min(nlas$gpstime) + 1e9, silent = TRUE)
  if (inherits(gps_time, "try-error") || (gps_time - 1e9) <= 604800) {
    date_str <- "XXXX-XX-XX"
  } else {
    date_time <- as.POSIXct("1980-01-06", tz = "UTC") + gps_time
    cx <- x_ul + tile_size/2; cy <- y_ul - tile_size/2
    tz <- try(lutz::tz_lookup(sf::st_as_sf(data.frame(x=cx,y=cy), coords=c("x","y"), crs=utm_crs)), silent = TRUE)
    if (inherits(tz, "try-error") || is.na(tz)) tz <- "UTC"
    date_str <- as.character(as.Date(as.POSIXct(date_time, tz = tz)))
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  out_tif <- file.path(output_dir, sprintf("%d_%d.tif", as.integer(x_ul), as.integer(y_ul)))
  terra::writeRaster(
    chm, filename = out_tif, datatype = "FLT4S",
    gdal = c("COMPRESS=DEFLATE", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"),
    overwrite = TRUE
  )
  return(paste(out_tif, "created"))
}

treelist_from_chm <- function(chm_tif, outdir = tempdir(), quiet = TRUE, min_crown_area=4, min_height=2, ...) {
  if (!file.exists(chm_tif)) stop("File not found: ", chm_tif)
  if (!requireNamespace("terra", quietly = TRUE)) stop("Package 'terra' is required.")
  if (!requireNamespace("cloud2trees", quietly = TRUE)) stop("Package 'cloud2trees' is required.")
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  r <- terra::rast(chm_tif)
  if (quiet) terra::sources(r)


'''
  added temdir local env reference to resolve the following Error:
    {"ok":false,"error":"Error in file.path(tempdir, \"tile_.tif\"): promise already under evaluation: recursive default argument reference or earlier problems?\n"}
'''
  print("Running raster2trees")
  tl <- cloud2trees::raster2trees(
    chm_rast = r,
    outfolder = outdir,
    min_height = min_height,
    min_crown_area = min_crown_area,
    tempdir = tempdir(),              # <--- this is what was changed
    ...)
  print("Ran raster2trees")

  # Always write a CSV with attributes only (no geometry)
  csv_path <- file.path(outdir, "treelist.csv")
  utils::write.csv(sf::st_drop_geometry(tl), csv_path, row.names = FALSE)

  return(tl)  # keep returning sf so caller can also write a GPKG, etc.
}

# ---------- CBH prediction utils ----------
treelist_add_cbh <- function(treelist_csv, cbh_model_rds, epsg) {
  if (!file.exists(treelist_csv)) stop("File not found: ", treelist_csv)
  if (!file.exists(cbh_model_rds)) stop("File not found: ", cbh_model_rds)

  # read inputs
  treelist <- readr::read_csv(treelist_csv, show_col_types = FALSE)
  cbh_mod  <- readr::read_rds(cbh_model_rds)

  # make treelist an sf object (keeps tree_x/tree_y)
  treelist <- treelist |>
    sf::st_as_sf(coords = c("tree_x", "tree_y"), crs = as.integer(epsg), remove = FALSE)

  # add coordinates and prep predictors
  coords <- sf::st_coordinates(treelist)
  treelist <- treelist |>
    dplyr::mutate(
      tree_y_zzz     = coords[,2],
      tree_x_zzz     = coords[,1],
      crown_area_zzz = .data$crown_area_m2
    )

  predictors <- treelist |>
    sf::st_drop_geometry() |>
    dplyr::select(dplyr::any_of(c("tree_y_zzz", "tree_x_zzz", "tree_height_m", "crown_area_zzz")))

  if (nrow(predictors) != nrow(treelist))
    stop("Predictor row count mismatch.")

  # predict CBH (support both randomForest::predict(newdata=) and tidymodels predict(new_data=))
  predicted_cbh <- tryCatch(
    predict(cbh_mod, newdata = predictors),
    error = function(e) predict(cbh_mod, new_data = predictors)
  )

  # coerce to numeric vector length nrow
  predicted_cbh <- as.numeric(predicted_cbh)[seq_len(nrow(treelist))]

  # add results
  treelist <- treelist |>
    dplyr::mutate(cbh_m = predicted_cbh)

  # overwrite the input CSV with new data (no geometry)
  out_cols <- intersect(
    c("index","treeID","tree_x","tree_y","tree_height_m","crown_area_m2","HMD_m","cbh_m"),
    names(sf::st_drop_geometry(treelist))
  )
  readr::write_csv(sf::st_drop_geometry(treelist)[, out_cols, drop = FALSE], treelist_csv)
  message("✅ Treelist with predicted CBH written to (overwritten): ", treelist_csv)

  return(treelist)
}

`%||%` <- function(a,b) if(!is.null(a)) a else b

# ---------- CLI (docopt) ----------
if (sys.nframe() == 0) {

  doc <- '
  Usage:
    Rutils.R treelist --chm=<file> [--outdir=<dir>] [--min_crown_area=<val>] [--min_height=<val>]
    Rutils.R chm --las=<file> --epsg=<code> [--outdir=<dir>]
    Rutils.R cbh --csv=<file> --model=<file> --epsg=<code> [--out=<file>]
    Rutils.R chm_create --las=<file> --epsg=<code> [--outdir=<dir>]
    Rutils.R (-h | --help)

  Options:
    --chm=<file>             Path to CHM GeoTIFF.
    --las=<file>             Path to input LAS/LAZ tile (expects "<x>_<y>_*.laz" naming).
    --csv=<file>             Path to treelist CSV with columns tree_x, tree_y, tree_height_m, crown_area_m2.
    --model=<file>           Path to CBH model RDS.
    --epsg=<code>            EPSG code for projected CRS (e.g., 26911).
    --outdir=<dir>           Output directory [default: /tmp].
    --out=<file>             Output CSV path.
    --min_crown_area=<val>   Minimum crown area in m² [default: 4].
    --min_height=<val>       Minimum height in meters [default: 2].
  '
  opt <- docopt::docopt(doc)

  tryCatch({
    if (isTRUE(opt$treelist)) {
      chm  <- opt$`--chm`
      outd <- opt$`--outdir` %||% tempdir()
      min_crown_area <- as.numeric(opt$`--min_crown_area`)
      min_height <- as.numeric(opt$`--min_height`)

      if (!dir.exists(outd)) dir.create(outd, recursive = TRUE, showWarnings = FALSE)
      print("Running treelist_from_chm")
      tl <- treelist_from_chm(chm, outdir = outd, min_crown_area=min_crown_area, min_height=min_height)
      print("Ran treelist_from_chm")

      gpkg <- file.path(outd, "treelist.gpkg")
      try(sf::st_write(tl, gpkg, layer = "crowns", delete_dsn = TRUE, quiet = TRUE), silent = TRUE)
      csv <- file.path(outd, "treelist.csv")
      cat(jsonlite::toJSON(list(ok=TRUE, csv=csv, gpkg=if (file.exists(gpkg)) gpkg else NULL, rows=nrow(tl)), auto_unbox=TRUE))

    } else if (isTRUE(opt$chm) || isTRUE(opt$chm_create)) {
      las   <- opt$`--las`
      epsg  <- as.integer(opt$`--epsg`)
      outd  <- opt$`--outdir` %||% "data"
      if (!dir.exists(outd)) dir.create(outd, recursive = TRUE, showWarnings = FALSE)

      msg <- chm_create(las, epsg, outd)
      cat(jsonlite::toJSON(list(ok=TRUE, message=msg), auto_unbox=TRUE))

    } else if (isTRUE(opt$cbh)) {
      csv   <- opt$`--csv`
      model <- opt$`--model`
      epsg  <- as.integer(opt$`--epsg`)
      tl <- treelist_add_cbh(csv, model, epsg)
      cat(jsonlite::toJSON(list(ok=TRUE, rows=nrow(tl)), auto_unbox=TRUE))
    }

  }, error=function(e){
    cat(jsonlite::toJSON(list(ok=FALSE, error=as.character(e)), auto_unbox=TRUE))
    quit(status=1)
  })
}


