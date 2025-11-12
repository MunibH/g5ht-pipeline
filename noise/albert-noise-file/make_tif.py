import glob
import os
import numpy as np
import tifffile

rfp_paths = glob.glob('rfp-noise-frames/*.tif')
gfp_paths = glob.glob('gfp-noise-frames/*.tif')

rfp = np.mean(np.array([tifffile.imread(i) for i in rfp_paths]), axis=0).astype(np.uint16)
gfp = np.mean(np.array([tifffile.imread(i) for i in gfp_paths]), axis=0).astype(np.uint16)

tifffile.imwrite('noise_111125.tif', np.array([gfp, rfp]), imagej=True)
