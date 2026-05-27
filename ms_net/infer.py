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
        map_location=device,            # <—
    ).to(device).eval()                 # <—

    val_dataloader = get_dataloader(net_dict, ['inf'], data_path=data_path, NORM_CONST=NORM_CONST)
    valdata = val_dataloader['inf'].dataset

    @torch.inference_mode()
    def get_ypred(dataset, sampleidx):
        sample, masks, xy = dataset[sampleidx]
        # print("sample:", sample)
        # print("sample type:", type(sample))

        # x is a list of tensors/arrays → ensure torch tensors on the same device
        x = [
            torch.as_tensor(wvsam, dtype=torch.float32, device=device).unsqueeze(0)
            for wvsam in xy[0]
        ]

        # move masks too (handles tensor, list/tuple, or None)
        if masks is not None:
            if isinstance(masks, (list, tuple)):
                masks = [torch.as_tensor(m, dtype=torch.float32, device=device) for m in masks]
            else:
                masks = torch.as_tensor(masks, dtype=torch.float32, device=device)

        # sanity check (catches device mismatches early)
        assert next(model.parameters()).device == x[0].device, \
            f"Model on {next(model.parameters()).device}, input on {x[0].device}"

        y_pred_list = model(x, masks)
        y_pred = y_pred_list[-1].detach().to('cpu')[0, 0, :, :]  # back to CPU for rasterio
        y_pred = y_pred * NORM_CONST  # de-normalize
        return y_pred

    # run predictions
    test_list = params.inf_list
    with open(test_list, 'r') as file:
        for i, line in enumerate(file):
            base_name = line.strip().split(".")[0]
            if trainSite is not None and testSite is not None:
                fileName = f'train_{trainSite}_test{testSite}_{base_name}.tif'
            else:    
                fileName = f'{base_name}.tif'

            print(f'base_name: {base_name}')

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
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

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



if __name__ == '__main__':
    # Example usage
    run_inference(
        data_path='/mnt/c/Users/zach/Desktop/canopy/sycanMarsh_INF_data/',
        site='sycanMarsh',
        NORM_CONST=46,
        model_loc='/mnt/c/Users/zach/Desktop/canopy/chm-ms-net/ms-net-architecture-mia/lightning_logs/FS_version_65/epoch-epoch=99.ckpt',
        epsg_code=32610
    )
