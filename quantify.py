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

colors = plt.cm.tab10(np.linspace(0, 1, 10))

# function to plot traces
def plot_traces(t, df, baseline_window=(0, 60), input_dir=None, time_type='min'):
    # time_type: 'min' or 'sec' or 'frame

    if time_type == 'min':
        t = t
        xlabel = 'Time (min)'
        xtick_step = 0.5
    elif time_type == 'sec':
        t = t * 60
        xlabel = 'Time (sec)'
        xtick_step = 30
    elif time_type == 'frame':
        t = np.arange(len(df))
        xlabel = 'Frame'
        xtick_step = 100
    else:
        raise ValueError('time_type must be "min", "sec", or "frame"')
    
    # get data as numpy array
    out = df.values
    # get column labels
    labels = df.columns.tolist()
    nlabels = out.shape[1]

    plt.figure(figsize=(10, 4))
    for i in range(nlabels):
        # plt.plot(t, out[:, i] / np.mean(out[:60, i]), label=labels[i], color=colors[i], lw=2)
        plt.plot(t, out[:, i] / np.mean(out[baseline_window[0]:baseline_window[1], i]), label=labels[i], color=colors[i], lw=2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    plt.xlabel(xlabel)
    plt.ylabel(r'$R/R_{baseline}$')
    plt.xlim(0, t[-1])
    plt.axhline(1, ls='--', c='k', zorder=0)
    # make the xaxis ticks dense (lots of values shown)
    # plt.xticks(np.arange(0, np.max(t)+1, step=0.5))
    plt.xticks(np.arange(0, t[-1], step=xtick_step))
    # rotate xticks 45 degrees
    plt.xticks(rotation=45)
    plt.tight_layout()
    if input_dir is not None:
        plt.savefig(os.path.join(input_dir, 'quantified.png'), dpi=300)
        plt.savefig(os.path.join(input_dir, 'quantified.svg'), dpi=300)
    plt.show()

def plot_rois(fixed, roi, input_dir=None):
    # visualize ROIs on max projection of fixed image   
    
    img = np.zeros((200, 500, 3), np.float32)
    img[..., 0] = np.max(fixed[:, 1], axis=0) # red channel max projection
    img[..., 0] = np.clip(img[..., 0] / 400, 0, 1) # adjust contrast for visualization
    img = (img * 255).astype(np.ubyte) # convert to uint8 for visualization
    # conver img to grayscale
    img = np.stack([img[..., 0]]*3, axis=-1)

    nlabels = np.max(roi)

    plt.figure(figsize=(10, 4))
    for i in range(nlabels):
        contours = measure.find_contours(np.max(roi == i + 1, axis=0), level=0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color=colors[i])
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    if input_dir is not None:
        plt.savefig(os.path.join(input_dir , 'roi.png'), dpi=300)
        plt.savefig(os.path.join(input_dir , 'roi.svg'), dpi=300)
    plt.show()

def main():

    input_dir = sys.argv[1]
    reg_dir = sys.argv[2]
    registered_dir = os.path.join(input_dir, reg_dir)
    baseline_window = sys.argv[3] if len(sys.argv) > 3 and isinstance(sys.argv[3], tuple) else (0, 60)
    plot_only = sys.argv[4] if len(sys.argv) > 4 else False
    time_type = sys.argv[5] if len(sys.argv) > 5 else 'min'

    tif_paths = glob.glob(os.path.join(registered_dir, '*.tif'))
    tif_paths = sorted(tif_paths)[:]

    # find fixed file
    fixed_fn = glob.glob(os.path.join(input_dir, 'fixed_[0-9][0-9][0-9][0-9]*.tif'))[0]
    fixed = tifffile.imread(fixed_fn)

    # load mask with metadata (it was saved to tif as tifffile.imwrite(roi_pth, roi.astype(np.uint8), imagej=True, metadata={'Labels': roi_labels}))
    roi = tifffile.imread(os.path.join(input_dir, 'roi.tif'))
    with tifffile.TiffFile(os.path.join(input_dir, 'roi.tif')) as tif:
        labels = tif.imagej_metadata['Labels']

    nlabels = len(labels)

    if not plot_only:
        out = np.zeros((len(tif_paths), nlabels))
        out[:] = np.nan
        for i in tqdm(range(len(out))):
            stack = tifffile.imread(tif_paths[i])
            for j in range(nlabels):
                denominator =  np.sum(stack[:, 1][roi == j + 1])
                if denominator > 0:
                    out[i, j] = np.sum(stack[:, 0][roi == j + 1]) / denominator

        t = np.arange(len(out)) * 0.533 / 60
        df = pd.DataFrame(out, index=t)
        df = df.interpolate()
        # save labeled columns
        df.columns = labels
        df.to_csv(os.path.join(input_dir, 'quantified.csv'))
        plot_traces(t, df, baseline_window=baseline_window, input_dir=input_dir, time_type=time_type)
        plot_rois(fixed, roi, input_dir=input_dir)
    else:
        df = pd.read_csv(os.path.join(input_dir, 'quantified.csv'), index_col=0)
        t = df.index.values
        plot_traces(t, df, baseline_window=baseline_window, input_dir=input_dir, time_type=time_type)
        plot_rois(fixed, roi, input_dir=input_dir)