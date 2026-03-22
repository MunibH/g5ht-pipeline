"""NIR worm segmentation, skeletonization, and eigenworm computation.

Provides a single entry point `process_eigenworms` that takes an H5 path and
returns per-frame tangent angle profiles, PCA eigenworms, and scores.
"""

import numpy as np
import torch
import h5py
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from skimage import morphology, measure
from skimage.morphology import skeletonize
import networkx as nx

import nir_utils
from segment.unet_model import UNet2D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = '/home/munib/code/g5ht-pipeline/segment/worm_segmentation_best_weights_0310.pt'
N_SPLINE_POINTS = 101
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(weights_path=DEFAULT_WEIGHTS, cuda_device=0):
    """Load UNet2D segmentation model and return (model, device)."""
    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(n_ch_in=1, n_class=1, n_feature=16, bilinear=False).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_frames(model, device, images):
    """Segment a stack of NIR images.

    Parameters
    ----------
    model : UNet2D model (eval mode).
    device : torch.device.
    images : (N, H, W) uint8 array of NIR frames.

    Returns
    -------
    seg : (N, H, W) uint8 binary masks.
    """
    n = len(images)
    img_float = images.astype(np.float32)
    # z-score normalise per-frame
    means = img_float.mean(axis=(1, 2), keepdims=True)
    stds = img_float.std(axis=(1, 2), keepdims=True) + 1e-8
    img_norm = (img_float - means) / stds

    seg = np.zeros((n, images.shape[1], images.shape[2]), dtype=np.uint8)

    for i in tqdm(range(0, n, BATCH_SIZE), desc='Segmenting batch'):
        batch = img_norm[i:i + BATCH_SIZE]
        tensor_in = torch.from_numpy(batch).unsqueeze(1).float().to(device)  # (B,1,H,W)
        with torch.no_grad():
            logits = model(tensor_in)
            prob = torch.sigmoid(logits)
            seg[i:i + len(batch)] = (prob.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)

    return seg


# ---------------------------------------------------------------------------
# Skeleton extraction (reuses graph-based approach from spline.py)
# ---------------------------------------------------------------------------

def _make_graph(ske):
    G = nx.Graph()
    rows, cols = np.where(ske)
    for r, c in zip(rows.tolist(), cols.tolist()):
        G.add_node((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ske.shape[0] and 0 <= nc < ske.shape[1] and ske[nr, nc]:
                G.add_edge((r, c), (nr, nc))
    return G


def _get_largest_path(G):
    nodes_to_remove = [n for n, d in G.degree() if d > 2]
    G.remove_nodes_from(nodes_to_remove)
    components = list(nx.connected_components(G))
    if not components:
        return []
    largest = max(components, key=len)
    sub = G.subgraph(largest)
    endpoints = [n for n, d in sub.degree() if d == 1]
    if len(endpoints) < 2:
        return list(sub.nodes())
    start = endpoints[0]
    ordered = [start] + [v for _, v in nx.bfs_edges(sub, source=start)]
    return ordered


def extract_skeleton(seg_frame):
    """Extract ordered skeleton coordinates from a single binary mask.

    Parameters
    ----------
    seg_frame : (H, W) binary mask.

    Returns
    -------
    coords : (M, 2) array of (row, col) ordered head-to-tail, or None if failed.
    """
    cleaned = morphology.remove_small_objects(seg_frame.astype(bool), max_size=99)
    cleaned = morphology.remove_small_holes(cleaned, max_size=100)
    ske = skeletonize(cleaned)
    G = _make_graph(ske)
    nodes = _get_largest_path(G)
    if len(nodes) < 10:
        return None
    return np.array(nodes)  # (M, 2) in (row, col)


# ---------------------------------------------------------------------------
# Spline resampling & angle computation
# ---------------------------------------------------------------------------

def resample_skeleton(coords, n_points=N_SPLINE_POINTS):
    """Resample ordered skeleton to n_points equally spaced along arc length.

    Parameters
    ----------
    coords : (M, 2) array in (row, col).

    Returns
    -------
    resampled : (n_points, 2) array in (row, col), or None if fitting fails.
    """
    if coords is None or len(coords) < 4:
        return None
    r, c = coords[:, 0].astype(float), coords[:, 1].astype(float)
    dists = np.sqrt(np.diff(r) ** 2 + np.diff(c) ** 2)
    cum = np.concatenate(([0], np.cumsum(dists)))
    total = cum[-1]
    if total < 1e-8:
        return None
    try:
        spl_r = UnivariateSpline(cum, r, s=0)
        spl_c = UnivariateSpline(cum, c, s=0)
    except Exception:
        return None
    steps = np.linspace(0, total, n_points)
    return np.column_stack([spl_r(steps), spl_c(steps)])


def compute_tangent_angles(resampled):
    """Compute mean-subtracted unwrapped tangent angle profile.

    Parameters
    ----------
    resampled : (n_points, 2) in (row, col).

    Returns
    -------
    angles : (n_points - 1,) array.
    """
    dr = np.diff(resampled[:, 0])
    dc = np.diff(resampled[:, 1])
    angles = np.arctan2(dc, dr)
    angles = np.unwrap(angles)
    angles -= np.mean(angles) # mean-subtract to remove global orientation differences
    
    # lower pass filter to remove high-frequency noise (optional, can be commented out)
    from scipy.signal import savgol_filter
    angles = savgol_filter(angles, window_length=11, polyorder=3, mode='nearest')
    
    return angles


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_eigenworms(h5_path, frames=None, n_pcs=4, cuda_device=0,
                       weights_path=DEFAULT_WEIGHTS, n_spline_points=N_SPLINE_POINTS):
    """Full pipeline: segment → skeleton → spline → angles → PCA eigenworms.

    Parameters
    ----------
    h5_path : str, path to the HDF5 file.
    frames : list of int or None.  If None, processes all frames.
    n_pcs : number of PCA components (eigenworms) to compute.
    cuda_device : CUDA device index.
    weights_path : path to UNet weights.
    n_spline_points : number of points to resample the spline to.

    Returns
    -------
    result : dict with keys
        'seg'             : (N, H, W) uint8 binary segmentation masks
        'skeletons'       : list of (M,2) arrays or None per frame
        'splines'         : list of (n_spline_points,2) arrays or None per frame
        'angles'          : list of (n_spline_points-1,) arrays or None per frame
        'angle_matrix'    : (N_valid, n_spline_points-1) matrix of valid angles
        'valid_mask'      : (N,) bool array
        'pca'             : fitted PCA object
        'pca_scores'      : (N_valid, n_pcs) PCA scores
        'eigenworms'      : (n_pcs, n_spline_points-1) eigenworm components
        'explained_var'   : (n_pcs,) explained variance ratio
        'frame_indices'   : list of int, the frame indices processed
    """
    # load images
    images = nir_utils.load_img_nir(h5_path, frames=frames)
    if images.ndim == 2:
        images = images[np.newaxis]  # single frame → (1,H,W)

    n = len(images)
    frame_indices = frames if isinstance(frames, list) else list(range(n))

    # segment
    model, device = load_model(weights_path, cuda_device)
    print(f'Segmenting {n} frames on {device}...')
    seg = segment_frames(model, device, images)

    # skeleton + spline + angles
    skeletons = []
    splines = []
    angles_list = []
    valid_mask = np.zeros(n, dtype=bool)

    for i in tqdm(range(n), desc='Skeletonizing'):
        skel = extract_skeleton(seg[i])
        skeletons.append(skel)
        if skel is None:
            splines.append(None)
            angles_list.append(None)
            continue
        rs = resample_skeleton(skel, n_spline_points)
        splines.append(rs)
        if rs is None:
            angles_list.append(None)
            continue
        ang = compute_tangent_angles(rs)
        angles_list.append(ang)
        valid_mask[i] = True

    # PCA on valid angle profiles
    angle_matrix = np.array([a for a in angles_list if a is not None])
    n_valid = len(angle_matrix)
    print(f'{n_valid}/{n} frames have valid angle profiles.')

    effective_pcs = min(n_pcs, n_valid, angle_matrix.shape[1]) if n_valid > 0 else 0
    if effective_pcs > 0:
        pca = PCA(n_components=effective_pcs)
        pca_scores = pca.fit_transform(angle_matrix)
        eigenworms = pca.components_
        explained_var = pca.explained_variance_ratio_
    else:
        pca = None
        pca_scores = np.empty((0, n_pcs))
        eigenworms = np.empty((0, n_spline_points - 1))
        explained_var = np.empty(0)

    return {
        'seg': seg,
        'skeletons': skeletons,
        'splines': splines,
        'angles': angles_list,
        'angle_matrix': angle_matrix,
        'valid_mask': valid_mask,
        'pca': pca,
        'pca_scores': pca_scores,
        'eigenworms': eigenworms,
        'explained_var': explained_var,
        'frame_indices': frame_indices,
    }
