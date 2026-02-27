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

# import matplotlib
# font = {'family' : 'Arial',
#         'weight' : 'normal',
#         'size'   : 15}
# matplotlib.rc('font', **font)
# plt.rcParams['svg.fonttype'] = 'none'
from utils import pretty_plot, default_plt_params
default_plt_params()

# Predefined hex colors for each ROI label (consistent across datasets)
LABEL_COLORS = {
    # long-form labels
    'procorpus':          '#F94144',
    'metacorpus':         '#FF9129',
    'isthmus':            '#F3BD3E',
    'terminal_bulb':      '#FFD981',
    'nerve_ring':         '#90BE6D',
    'ventral_nerve_cord': '#43AA8B',
    'dorsal_nerve_cord':  '#000000',
    # short-form labels
    'PC':                 '#F94144',
    'MC':                 '#FF9129',
    'IM':                 '#F3BD3E',
    'TB':                 '#FFD981',
    'NR':                 '#90BE6D',
    'VNC':                '#43AA8B',
    'DNC':                '#000000',
}

_FALLBACK_COLORS = ['#17becf', '#bcbd22', '#7f7f7f', '#aec7e8', '#ffbb78', '#98df8a']

def get_label_color(label, fallback_idx=0):
    """Return the hex color for a given label, with fallback for unknown labels."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    return _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]

# function to plot traces
def plot_traces(t, df, baseline_window=(0, 60), input_dir=None, time_type='min', encounter_value=None, normalize_to_max=False):
    # time_type: 'min' or 'sec' or 'frame
    # if encounter_value is not None, plot a vertical shaded region (-1 to +1 min around encounter_value) to indicate encounter time (plot at 0)
    # then subtract encounter_value from t to align traces to encounter time (plot at 0)
    # encounter_value should be in the same units as time_type

    lw = 2.5

    if time_type == 'min':
        t = t
        xlabel = 'Time (min)'
        xtick_step = 0.5
    elif time_type == 'sec':
        t = t * 60
        xlabel = 'Time (sec)'
        xtick_step = 50
    elif time_type == 'frame':
        t = np.arange(len(df))
        xlabel = 'Frame'
        xtick_step = 100
    else:
        raise ValueError('time_type must be "min", "sec", or "frame"')
    
    if encounter_value is not None:
        t = t - encounter_value
    
    # get data as numpy array
    out = df.values
    # get column labels
    labels = df.columns.tolist()
    nlabels = out.shape[1]

    # sort labels, and corresponding data and colors, by label name
    sorted_indices = np.argsort(labels)
    labels = [labels[i] for i in sorted_indices]
    out = out[:, sorted_indices]

    fig, ax = pretty_plot(figsize=(12, 3.5))
    # plt.figure(figsize=(10, 4))
    fallback_idx = 0
    for i in range(nlabels):
        c = get_label_color(labels[i], fallback_idx)
        if labels[i] not in LABEL_COLORS:
            fallback_idx += 1
        r_baseline = np.mean(out[baseline_window[0]:baseline_window[1], i])
        r_over_baseline = out[:, i] / r_baseline
        r_over_baseline_normed = r_over_baseline / np.max(r_over_baseline)
        if normalize_to_max:
            ax.plot(t, r_over_baseline_normed, label=labels[i], color=c, lw=lw)
        else:
            ax.plot(t, r_over_baseline, label=labels[i], color=c, lw=lw)
    
    if encounter_value is not None:
        ax.axvspan(-3, 3, color='gray', alpha=0.3, label='Encounter')    
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    if encounter_value is not None:
        ax.set_xlabel(f'{xlabel} from encounter')
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$R/R_{baseline}$')
    if encounter_value is not None:
        ax.set_xlim(-encounter_value, t[-1])
    else:
        ax.set_xlim(0, t[-1])
    ax.axhline(1, ls='--', c='k', zorder=0)
    # make the xaxis ticks dense (lots of values shown)
    # ax.set_xticks(np.arange(0, np.max(t)+1, step=0.5))
    # ax.set_xticks(np.arange(0, t[-1], step=xtick_step))
    # rotate xticks 45 degrees
    # ax.set_xticklabels(ax.get_xticks(), rotation=45)
    plt.tight_layout()
    if input_dir is not None:
        plt.savefig(os.path.join(input_dir, 'quantified.png'), dpi=300)
        plt.savefig(os.path.join(input_dir, 'quantified.svg'), dpi=300)
    plt.show()

def plot_rois(fixed, roi, labels=None, input_dir=None):
    # visualize ROIs on max projection of fixed image   
    
    img = np.zeros((200, 500, 3), np.float32)
    img[..., 0] = np.max(fixed[:, 1], axis=0) # red channel max projection
    img[..., 0] = np.clip(img[..., 0] / 400, 0, 1) # adjust contrast for visualization
    img = (img * 255).astype(np.ubyte) # convert to uint8 for visualization
    # conver img to grayscale
    img = np.stack([img[..., 0]]*3, axis=-1)

    nlabels = np.max(roi)

    plt.figure(figsize=(10, 4))
    fallback_idx = 0
    for i in range(nlabels):
        if labels is not None and i < len(labels):
            c = get_label_color(labels[i], fallback_idx)
            if labels[i] not in LABEL_COLORS:
                fallback_idx += 1
        else:
            c = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        contours = measure.find_contours(np.max(roi == i + 1, axis=0), level=0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color=c)
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
    encounter_value = sys.argv[6] if len(sys.argv) > 6 else None
    normalize_to_max = sys.argv[7] if len(sys.argv) > 7 else False

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
        # remove column with title 'dorsal_nerve_cord' from df if it exists
        if 'dorsal_nerve_cord' in df.columns:
            df = df.drop(columns=['dorsal_nerve_cord'])
        plot_traces(t, df, baseline_window=baseline_window, input_dir=input_dir, time_type=time_type, encounter_value=encounter_value, normalize_to_max=normalize_to_max)
        # remove dorsal_nerve_cord from labels if it exists
        if 'dorsal_nerve_cord' in labels:
            labels = [label for label in labels if label != 'dorsal_nerve_cord']
        plot_rois(fixed, roi, labels=labels, input_dir=input_dir)
    else:
        df = pd.read_csv(os.path.join(input_dir, 'quantified.csv'), index_col=0)
        t = df.index.values
        # remove column with title 'dorsal_nerve_cord' from df if it exists
        if 'dorsal_nerve_cord' in df.columns:
            df = df.drop(columns=['dorsal_nerve_cord'])
        plot_traces(t, df, baseline_window=baseline_window, input_dir=input_dir, time_type=time_type, encounter_value=encounter_value, normalize_to_max=normalize_to_max)
        # remove dorsal_nerve_cord from labels if it exists
        if 'dorsal_nerve_cord' in labels:
            labels = [label for label in labels if label != 'dorsal_nerve_cord']
        plot_rois(fixed, roi, labels=labels, input_dir=input_dir)