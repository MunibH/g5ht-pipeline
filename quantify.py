import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
from skimage import measure
import os
from tqdm import tqdm
import scipy.ndimage as ndi

import matplotlib
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

def main():

    input_dir = sys.argv[1]
    registered_dir = os.path.join(input_dir, 'registered')

    tif_paths = glob.glob(os.path.join(registered_dir, '*.tif'))
    tif_paths = sorted(tif_paths)[:]
    mask = tifffile.imread(os.path.join(input_dir, 'roi.tif'))
    mask = ndi.zoom(mask, zoom=(1/3,1,1), order=0) # ensure mask has same shape as stacks

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
    plt.savefig(os.path.join(input_dir, 'quantified.svg'), dpi=300)
    plt.show()

    fixed = tifffile.imread(os.path.join(input_dir, 'fixed.tif'))
    img = np.zeros((200, 500, 3), np.float32)
    img[..., 0] = np.max(fixed[:, 1], axis=0) # green channel max projection
    img[..., 0] = np.clip(img[..., 0] / 400, 0, 1) # adjust contrast for visualization
    img = (img * 255).astype(np.ubyte) # convert to uint8 for visualization

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
    plt.savefig(os.path.join(input_dir , 'roi.svg'), dpi=300)
    plt.show()

    # -- separate channels --
    # out_r = np.zeros((len(tif_paths), 3))
    # out_r[:] = np.nan
    # out_g = out_r.copy()
    # for i in tqdm(range(len(out_r))):
    #     stack = tifffile.imread(tif_paths[i])
    #     for j in range(3):
    #         out_g[i, j] = np.sum(stack[:, 0][mask == j + 1])
    #         out_r[i, j] = np.sum(stack[:, 1][mask == j + 1])

    # t = np.arange(len(out_r)) * 0.533 / 60
    # df_r = pd.DataFrame(out_r, index=t)
    # df_g = pd.DataFrame(out_g, index=t)
    # df_r = df_r.interpolate()
    # df_g = df_g.interpolate()
    # df_r.to_csv(os.path.join(input_dir, 'quantified_red.csv'))
    # df_g.to_csv(os.path.join(input_dir, 'quantified_green.csv'))

    # plt.figure(figsize=(10, 4))
    # plt.plot(t, out_r[:, 0] / np.mean(out_r[:60, 0]), label='Dorsal nerve ring')
    # plt.plot(t, out_r[:, 1] / np.mean(out_r[:60, 1]), label='Ventral nerve ring')
    # plt.plot(t, out_r[:, 2] / np.mean(out_r[:60, 2]), label='Pharynx')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    # plt.xlabel('Time (min)')
    # plt.ylabel(r'$F/F_{baseline}$')
    # plt.xlim(0, np.max(t))
    # plt.axhline(1, ls='--', c='k', zorder=0)
    # plt.tight_layout()
    # plt.savefig(os.path.join(input_dir, 'quantified_red.png'), dpi=300)
    # plt.show()

    # plt.figure(figsize=(10, 4))
    # plt.plot(t, out_g[:, 0] / np.mean(out_g[:60, 0]), label='Dorsal nerve ring')
    # plt.plot(t, out_g[:, 1] / np.mean(out_g[:60, 1]), label='Ventral nerve ring')
    # plt.plot(t, out_g[:, 2] / np.mean(out_g[:60, 2]), label='Pharynx')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    # plt.xlabel('Time (min)')
    # plt.ylabel(r'$F/F_{baseline}$')
    # plt.xlim(0, np.max(t))
    # plt.axhline(1, ls='--', c='k', zorder=0)
    # plt.tight_layout()
    # plt.savefig(os.path.join(input_dir, 'quantified_green.png'), dpi=300)
    # plt.show()
