# ==============================================================================
# visualize_treelist.R
# Plot tree heights from a crown polygon file (e.g., final_detected_crowns.gpkg),
# optionally for a subset of the area, and save the plot as a PNG.
#
# Run in RStudio:  open the file, edit Section 1, click "Source"
# Run in terminal: Rscript visualize_treelist.R [input_file] [output_png]
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. USER INPUTS  (edit this section only)
# ------------------------------------------------------------------------------

# --- files ---
input_file <- "/project/wildfirehydro/ltiede/CHM_2/fs_1k_INF_data/pred/final_detected_crowns.gpkg"
output_png <- "/project/wildfirehydro/ltiede/CHM_2/treelist_metrics/z_tree_heights.png"

# --- column to plot ---
height_col <- "tree_height_m"

# --- subset mode: choose ONE of
#     "none"    plot everything
#     "height"  keep trees within a height range
#     "window"  keep trees whose location falls in an x/y window (crowns stay whole)
#     "crop"    clip crowns to an x/y rectangle (crowns are cut at the edge)
#     "aoi"     keep crowns touching a polygon file
#     "circle"  keep crowns touching a circle around a point
subset_mode <- "none"

# --- settings for "height" ---
# min_height <- 5       # meters
# max_height <- Inf     # meters (Inf = no upper limit)
min_height <- 5       # meters
max_height <- 50     # meters (Inf = no upper limit)

# --- settings for "window" and "crop" (in the file's CRS units) ---
xmin <- 500100
xmax <- 500300
ymin <- 4300200
ymax <- 4300400

# --- settings for "aoi" ---
aoi_file <- "path/to/aoi.gpkg"

# --- settings for "circle" (center in the file's CRS units; radius in CRS units) ---
circle_x      <- 500200
circle_y      <- 4300300
circle_radius <- 50

# --- plot appearance ---
palette            <- "Oranges"  # any RColorBrewer sequential palette name
legend_title       <- "tree ht. (m)"
shared_color_scale <- TRUE       # TRUE = colors match the full dataset's height range,
                                 # so subsets are comparable to the full-area plot

# --- output size ---
png_width  <- 8      # inches
png_height <- 8      # inches
png_dpi    <- 300


# ------------------------------------------------------------------------------
# 2. SETUP
# ------------------------------------------------------------------------------

# optional command-line overrides: Rscript visualize_treelist.R in.gpkg out.png
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) input_file <- args[1]
if (length(args) >= 2) output_png <- args[2]

# check required packages
required <- c("sf", "dplyr", "ggplot2")
missing_pkgs <- required[!vapply(required, requireNamespace,
                                 logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop("Missing packages: ", paste(missing_pkgs, collapse = ", "),
       "\nInstall with: install.packages(c(",
       paste0('"', missing_pkgs, '"', collapse = ", "), "))",
       call. = FALSE)
}

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(ggplot2)
})


# ------------------------------------------------------------------------------
# 3. READ AND CHECK DATA
# ------------------------------------------------------------------------------

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file, call. = FALSE)
}

crowns_sf <- st_read(input_file, quiet = TRUE)

if (!height_col %in% names(crowns_sf)) {
  stop("Column '", height_col, "' not found. Available columns: ",
       paste(names(crowns_sf), collapse = ", "), call. = FALSE)
}

message("Read ", nrow(crowns_sf), " features from: ", input_file)
message("Geometry type: ", paste(unique(as.character(st_geometry_type(crowns_sf))),
                                 collapse = ", "))
message("CRS: ", st_crs(crowns_sf)$input)

# full-dataset height range, used for a consistent color scale
full_range <- range(crowns_sf[[height_col]], na.rm = TRUE)


# ------------------------------------------------------------------------------
# 4. SUBSET
# ------------------------------------------------------------------------------

# helper: x/y of each tree (uses tree_x/tree_y if present, else crown centroids)
get_xy <- function(x) {
  if (all(c("tree_x", "tree_y") %in% names(x))) {
    data.frame(x = x$tree_x, y = x$tree_y)
  } else {
    xy <- st_coordinates(suppressWarnings(st_centroid(st_geometry(x))))
    data.frame(x = xy[, 1], y = xy[, 2])
  }
}

sub_sf <- switch(
  subset_mode,

  "none" = crowns_sf,

  "height" = crowns_sf %>%
    filter(.data[[height_col]] >= min_height,
           .data[[height_col]] <= max_height),

  "window" = {
    xy <- get_xy(crowns_sf)
    crowns_sf[xy$x >= xmin & xy$x <= xmax &
              xy$y >= ymin & xy$y <= ymax, ]
  },

  "crop" = {
    bb <- st_bbox(c(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
                  crs = st_crs(crowns_sf))
    st_agr(crowns_sf) <- "constant"
    suppressWarnings(st_crop(crowns_sf, bb))
  },

  "aoi" = {
    if (!file.exists(aoi_file)) stop("AOI file not found: ", aoi_file, call. = FALSE)
    aoi <- st_read(aoi_file, quiet = TRUE) %>% st_transform(st_crs(crowns_sf))
    st_filter(crowns_sf, aoi)
  },

  "circle" = {
    pt <- st_sfc(st_point(c(circle_x, circle_y)), crs = st_crs(crowns_sf))
    st_filter(crowns_sf, st_buffer(pt, dist = circle_radius))
  },

  stop("Unknown subset_mode: '", subset_mode,
       "'. Use one of: none, height, window, crop, aoi, circle.", call. = FALSE)
)

if (nrow(sub_sf) == 0) {
  b <- st_bbox(crowns_sf)
  stop("No trees remain after subsetting (mode = '", subset_mode, "').\n",
       "Full data extent is: xmin=", round(b["xmin"]), " xmax=", round(b["xmax"]),
       " ymin=", round(b["ymin"]), " ymax=", round(b["ymax"]), call. = FALSE)
}

message("Plotting ", nrow(sub_sf), " trees (subset mode: ", subset_mode, ")")


# ------------------------------------------------------------------------------
# 5. PLOT
# ------------------------------------------------------------------------------

plt_ht <-
  sub_sf %>%
  ggplot(aes(fill = .data[[height_col]])) +
  geom_sf() +
  scale_fill_distiller(
    palette   = palette,
    name      = legend_title,
    direction = 1,
    limits    = if (shared_color_scale) full_range else NULL
  ) +
  theme_void() +
  theme(legend.position = "top", legend.direction = "horizontal")

# show in RStudio / interactive sessions
print(plt_ht)


# ------------------------------------------------------------------------------
# 6. SAVE
# ------------------------------------------------------------------------------

ggsave(output_png, plt_ht,
       width = png_width, height = png_height, dpi = png_dpi, bg = "white")

message("Saved plot to: ", normalizePath(output_png))