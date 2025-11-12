import glob
import os
import numpy as np
import tifffile as tif
from nd2reader import ND2Reader
from tqdm import tqdm
from joblib import Parallel, delayed

nd2s = glob.glob('/orcd/data/flavell/001/g5ht/*/*/*.nd2')
os.system('rm -rf gfp-noise-frames')
os.system('rm -rf rfp-noise-frames')
print(f'Found {len(nd2s)} nd2 files')

def save_noise_frames(nd2, gfp_lim=1.067e8, rfp_lim=1.074e8):
    name = os.path.splitext(os.path.basename(nd2))[0]
    os.makedirs(f'gfp-noise-frames', exist_ok=True)
    os.makedirs(f'rfp-noise-frames', exist_ok=True)
    with ND2Reader(nd2) as f:
        frame_range = f.metadata['frames']
        for i in tqdm(frame_range):
            gfp, rfp = f.get_frame_2D(0, i), f.get_frame_2D(1, i)
            gfp_sum, rfp_sum = np.sum(gfp), np.sum(rfp)
            if gfp_sum < gfp_lim:
                tif.imwrite(f'gfp-noise-frames/{name}_{i:05d}.tif', gfp)
            if rfp_sum < rfp_lim:
                tif.imwrite(f'rfp-noise-frames/{name}_{i:05d}.tif', rfp)
    print('Done with {name}!')

Parallel(n_jobs=-1)(delayed(save_noise_frames)(nd2) for nd2 in nd2s)
