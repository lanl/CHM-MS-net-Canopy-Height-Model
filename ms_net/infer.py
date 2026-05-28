"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

import os
import re
import torch
import rasterio
from rasterio.transform import from_origin

try: 
    from .network_2D_lightning import MS_Net
    from .ms_parser import parse_args
    from .pore_utils_2D import get_dataloader
except (ImportError, ModuleNotFoundError):
    from network_2D_lightning import MS_Net
    from ms_parser import parse_args
    from pore_utils_2D import get_dataloader


def run_inference(data_path, site, NORM_CONST, model_loc, epsg_code, phase='inf', trainSite=None, testSite=None):
    """Run CHM predictions and save results as GeoTIFFs."""
    output_folder = os.path.join(data_path, 'chm_preds')
    os.makedirs(output_folder, exist_ok=True)  # Create early

    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = parse_args(['--data_loc', data_path])
    params.inf_list = os.path.join(data_path, f'{site}_inflist.txt')
    if phase == 'test':
        params.inf_list = os.path.join(data_path, f'{site}_testlist.txt')

    params.x_array = ['wvimg', 'solar', 'sensor', 'dem']
    params.y_array = ['chm']
    params.x_xform = [None, None, None, None]
    params.y_xform = [None]
    params.c_xform = [None]
    params.model_loc = 'chks'

    net_dict = vars(params)
    net_dict['uz_stats'] = {'scalar': 1}
    net_dict['edist_stats'] = {'scalar': 2.7}
    net_dict['c_stats'] = {'scalar': 1}
    net_dict['p_stats'] = {'scalar': 0}
    net_dict['D_stats'] = {'scalar': 0}

    # --- load model ONTO the chosen device
    model = MS_Net.load_from_checkpoint(
        model_loc,
        net_name='FireNet',
        num_scales=3,
        num_features=4,
        num_filters=8,
        f_mult=4,
        map_location=device,
    ).to(device).eval()

    val_dataloader = get_dataloader(net_dict, ['inf'], data_path=data_path, NORM_CONST=NORM_CONST)
    valdata = val_dataloader['inf'].dataset

    @torch.inference_mode()
    def get_ypred(dataset, sampleidx):
        sample, masks, xy = dataset[sampleidx]

        x = [
            torch.as_tensor(wvsam, dtype=torch.float32, device=device).unsqueeze(0)
            for wvsam in xy[0]
        ]

        if masks is not None:
            if isinstance(masks, (list, tuple)):
                masks = [torch.as_tensor(m, dtype=torch.float32, device=device) for m in masks]
            else:
                masks = torch.as_tensor(masks, dtype=torch.float32, device=device)

        assert next(model.parameters()).device == x[0].device, \
            f"Model on {next(model.parameters()).device}, input on {x[0].device}"

        y_pred_list = model(x, masks)
        y_pred = y_pred_list[-1].detach().to('cpu')[0, 0, :, :]
        y_pred = y_pred * NORM_CONST
        
        # 🔥 FIX 1: Clear GPU memory after each prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return y_pred

    # 🔥 FIX 2: Read test list FIRST, then close the file before processing
    test_list = params.inf_list
    with open(test_list, 'r') as file:
        test_files = [line.strip() for line in file if line.strip()]
    
    # 🔥 FIX 3: Add error tracking
    successful_writes = []
    failed_writes = []
    
    # run predictions
    for i, line in enumerate(test_files):
        base_name = line.split(".")[0]
        if trainSite is not None and testSite is not None:
            fileName = f'train_{trainSite}_test{testSite}_{base_name}.tif'
        else:    
            fileName = f'{base_name}.tif'

        print(f'Processing {i+1}/{len(test_files)}: {base_name}')

        try:
            y_pred = get_ypred(valdata, i)

            # parse coordinates from filename
            match = re.search(r'_(\d+)_([\d]+)$', base_name)
            if not match:
                raise ValueError(f"Filename {base_name} does not match expected format x_y")
            easting, northing = int(match.group(1)), int(match.group(2))

            # CHM prediction grid (upper-left origin assumed)
            pixel_size = 0.5
            transform = from_origin(easting, northing, pixel_size, pixel_size)

            output_path = os.path.join(output_folder, fileName)

            # 🔥 FIX 4: Write with explicit flushing
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=y_pred.shape[0],
                width=y_pred.shape[1],
                count=1,
                dtype='float32',
                crs=f'EPSG:{epsg_code}',
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(y_pred.numpy(), 1)
                # Explicitly flush to disk
            
            # 🔥 FIX 5: Verify file was written correctly
            if not os.path.exists(output_path):
                raise IOError(f"File was not created: {output_path}")
            
            file_size = os.path.getsize(output_path)
            if file_size < 156:  # TIFF header minimum
                raise IOError(f"File too small ({file_size} bytes), likely corrupted")
            
            successful_writes.append(fileName)
            
        except Exception as e:
            print(f"  ❌ Error processing {base_name}: {e}")
            failed_writes.append((fileName, str(e)))
            # Continue with next file instead of crashing
            continue
    
    # 🔥 FIX 6: Final sync to ensure all writes are complete
    import subprocess
    try:
        subprocess.run(['sync'], check=False, timeout=5)  # Linux/Mac
    except:
        pass  # Windows doesn't have sync, that's ok
    
    # 🔥 FIX 7: Report results
    print(f"\n✓ Successfully wrote {len(successful_writes)} files")
    if failed_writes:
        print(f"❌ Failed to write {len(failed_writes)} files:")
        for fname, error in failed_writes:
            print(f"  - {fname}: {error}")
    
    return successful_writes, failed_writes  # Return results for caller to check



if __name__ == '__main__':
    # Example usage
    run_inference(
        data_path='/mnt/c/Users/zach/Desktop/canopy/sycanMarsh_INF_data/',
        site='sycanMarsh',
        NORM_CONST=46,
        model_loc='/mnt/c/Users/zach/Desktop/canopy/chm-ms-net/ms-net-architecture-mia/lightning_logs/FS_version_65/epoch-epoch=99.ckpt',
        epsg_code=32610
    )
