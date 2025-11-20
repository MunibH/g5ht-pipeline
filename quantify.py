import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
from skimage import measure
import os
from tqdm import tqdm

import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)


def main():

    input_dir = sys.argv[1]
    registered_dir = os.path.join(input_dir, 'registered')

    tif_paths = glob.glob(os.path.join(registered_dir, '*.tif'))
    tif_paths = sorted(tif_paths)[:]
    mask = tifffile.imread(os.path.join(input_dir, 'roi.tif'))

    out = np.zeros((len(tif_paths), 3))
    out[:] = np.nan
    for i in tqdm(range(len(out))):
        stack = tifffile.imread(tif_paths[i])
        for j in range(3):
            denominator =  np.sum(stack[:, 1][mask == j + 1])
            if denominator > 0:
                out[i, j] = np.sum(stack[:, 0][mask == j + 1]) / denominator

    t = np.arange(len(out)) * 0.533 / 60
    df = pd.DataFrame(out, index=t)
    df = df.interpolate()
    df.to_csv(os.path.join(input_dir, 'quantified.csv'))

    plt.figure(figsize=(10, 4))
    plt.plot(t, out[:, 0] / np.mean(out[:60, 0]), label='Dorsal nerve ring')
    plt.plot(t, out[:, 1] / np.mean(out[:60, 1]), label='Ventral nerve ring')
    plt.plot(t, out[:, 2] / np.mean(out[:60, 2]), label='Pharynx')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    plt.xlabel('Time (min)')
    plt.ylabel(r'$F/F_{baseline}$')
    plt.xlim(0, np.max(t))
    plt.axhline(1, ls='--', c='k', zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, 'quantified.png'), dpi=300)
    plt.show()

    fixed = tifffile.imread(os.path.join(input_dir, 'fixed.tif'))
    img = np.zeros((200, 500, 3), np.float32)
    img[..., 0] = np.max(fixed[:, 1], axis=0)
    img[..., 0] = np.clip(img[..., 0] / 400, 0, 1)
    img = (img * 255).astype(np.ubyte)

    plt.figure(figsize=(10, 4))
    contours = measure.find_contours(np.max(mask == 1, axis=0), level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color='C0', linewidth=2)
    contours = measure.find_contours(np.max(mask == 2, axis=0), level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color='C1', linewidth=2)
    contours = measure.find_contours(np.max(mask == 3, axis=0), level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color='C2', linewidth=2)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir , 'roi.png'), dpi=300)
    plt.show()
