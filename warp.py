import numpy as np
from skimage import io, transform
from scipy.interpolate import splprep, splev
import tifffile
import json
import sys
import os
from joblib import Parallel, delayed

def get_length(tck):
    u_fine = np.linspace(0, 1, 100)
    x_fine, y_fine = splev(u_fine, tck)
    dx, dy = np.diff(x_fine), np.diff(y_fine)
    return np.sum(np.sqrt(dx ** 2 + dy ** 2))

def get_pts(tck, ts, interval=10, r=100):
    dxdt, dydt = splev(ts, tck, der=1)
    x, y = splev(ts, tck, der=0)
    out = []
    for i in range(len(ts)):
        norm = np.sqrt(dydt[i] ** 2 + dxdt[i] ** 2)
        dx, dy = dxdt[i] / norm, dydt[i] / norm
        for j in np.arange(-r, r + interval, interval):
            pt = (x[i] + dy * j, y[i] -dx * j)
            out.append((pt[1], pt[0]))
    moving_pts = np.array(out)
    return moving_pts

def initialize_tform(spline_data, nose_len=100, body_len=400):
    tck, u = splprep(spline_data.T)
    l = get_length(tck)
    t_min = -nose_len / l
    t_max = body_len / l
    u_plot = np.linspace(t_min, t_max, 51)
    pts = get_pts(tck, u_plot)
    src_cols = np.linspace(0, 500, 51)
    src_rows = np.linspace(0, 200, 21)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src, pts)
    return tform

def mask_warp(mask, form):
    return transform.warp(mask, tform, output_shape=(200, 500), order=0)

# load spline pts and initialize transform
def main():
    out_dir, index = sys.argv[1], int(sys.argv[2])
    with open(out_dir + '/followed.json', 'r') as f:
        spline_dict = json.load(f)
    spline_dict = {int(k): v for k, v in spline_dict.items()}
    spline_data = np.array(spline_dict[index])
    tform = initialize_tform(spline_data)
    print('Done initalizing transform!')

    # load stack and warp in parallel
    stack = tifffile.imread(f'{out_dir}/tif/{index:04d}.tif').astype(np.float32)
    def slice_warp(index):
        frame = stack[index]
        warp = lambda chn_frame: transform.warp(chn_frame, tform, output_shape=(200, 500), preserve_range=True, order=3)
        out = np.array((warp(frame[0]), warp(frame[1])))
        return np.clip(out, 0, 4095).astype(np.uint16)
    warped = Parallel(n_jobs=-1)(delayed(slice_warp)(i) for i in range(len(stack)))
    warped = np.array(warped)
    os.makedirs(out_dir + '/warped', exist_ok=True)
    tifffile.imwrite(f'{out_dir}/warped/{index:04d}.tif', warped, imagej=True)
    print('Done warping stack!')

    # load mask and warp
    dilated = tifffile.imread(f'{out_dir}/dilated.tif')
    mask = dilated[index]
    warped_mask = transform.warp(mask, tform, output_shape=(200, 500), order=0)
    os.makedirs(out_dir + '/masks', exist_ok=True)
    tifffile.imwrite(f'{out_dir}/masks/{index:04d}.tif', warped_mask)
    print('Done warping mask!')

if __name__ == '__main__':
    main()
