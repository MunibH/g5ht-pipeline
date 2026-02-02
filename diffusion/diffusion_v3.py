"""
diffusion_v3.py – Point Source Reaction–Diffusion Model
========================================================

KEY DIFFERENCE FROM diffusion_v2:
---------------------------------
This version uses **fixed point source locations** rather than learning
spatially-extended sources via NMF. This addresses the identifiability problem
where NMF sources become "pre-diffused" and leave no variance for the 
diffusion term to explain, causing D → 0.

APPROACH:
---------
1. **Identify source locations**: Use heuristics (variance, early response,
   peak detection, spatial gradients) to find candidate source pixels.

2. **Point source representation**: Each source is a delta function at a
   specific (row, col) location. The source field becomes:
       s(x,y,t) = Σᵢ aᵢ(t) · δ(x - xᵢ, y - yᵢ)
   where (xᵢ, yᵢ) are the identified source locations.

3. **Optimization**: Only optimize for:
   - a(t): Source time courses (T, M) - when each source fires
   - D: Diffusion coefficient (pixels²/s)
   - k: Decay rate (1/s)
   - g: Global source gain

4. **Identifiability**: Because sources are spatially localized (delta functions),
   the diffusion term MUST explain observed spatial spreading. This breaks
   the degeneracy that plagued the NMF approach.

MATHEMATICAL MODEL:
-------------------
    di/dt = D * ∇²i - k * i + g * Σₘ aₘ(t) * δ(x - xₘ, y - yₘ)

Discretized on the graph Laplacian with backward Euler time stepping.

Author: Munib Hasnain
Date: 2026-01-26
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Union, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter, laplace
from scipy.signal import find_peaks
from scipy.optimize import minimize, Bounds

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Configure module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PointSourceResult:
    """Container for point source diffusion model results."""
    
    # Fitted parameters
    D: float                              # Diffusion coefficient (pixels²/s)
    k: float                              # Decay rate (1/s)
    g: float                              # Source gain
    
    # Source information
    source_locations: np.ndarray          # (M, 2) array of (row, col) coordinates
    source_node_indices: np.ndarray       # (M,) node indices in Laplacian graph
    a_t: np.ndarray                       # (T, M) source time courses
    n_sources: int                        # Number of sources M
    
    # Data used
    Y_used: np.ndarray                    # (T, H, W) movie used for fitting
    mask_used: np.ndarray                 # (H, W) mask used
    
    # Reconstruction
    Ihat_masked: np.ndarray               # (T, N) reconstruction on masked pixels
    mse: float                            # Mean squared error
    r_squared: float                      # Coefficient of determination
    
    # Metadata
    dt: float                             # Time step (seconds)
    bin_factors: Tuple[int, int]          # Spatial binning factors
    
    # Graph structure
    L: sp.csr_matrix = field(repr=False)
    pix_to_node: np.ndarray = field(repr=False)
    node_to_pix: np.ndarray = field(repr=False)
    
    # Optimization details
    opt_result: Any = field(default=None, repr=False)
    source_detection_method: str = "combined"


@dataclass
class SourceDetectionConfig:
    """Configuration for point source detection."""
    
    n_sources: int = 8                    # Number of sources to detect per method
    methods: Tuple[str, ...] = ('variance', 'early_response', 'peak_detection', 'gradient')
    early_window: Tuple[int, int] = (10, 60)   # Frames for early response detection
    baseline_window: Tuple[int, int] = (0, 30) # Frames for baseline computation
    spatial_exclusion_radius: int = 5     # Min distance between sources (pixels)
    peak_prominence: float = 0.3          # Prominence threshold for peak detection
    peak_distance: int = 10               # Min frames between peaks
    combine_method: str = 'intersection'  # 'intersection', 'union', 'variance' 
    final_n_sources: Optional[int] = None # If set, limit final sources to this number


# =============================================================================
# 1) POINT SOURCE IDENTIFICATION
# =============================================================================

def identify_point_sources(
    Y: np.ndarray, 
    mask: np.ndarray, 
    config: Optional[SourceDetectionConfig] = None,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Identify candidate point source locations from fluorescence movie.
    
    Uses multiple heuristics to find pixels that are likely to be sources:
    - Temporal variance: high variance = active site
    - Early response: first pixels to respond = sources
    - Peak detection: pixels with transient peaks = burst-like release
    - Spatial gradient: local maxima in mean intensity = source centers
    
    Parameters
    ----------
    Y : (T, H, W) array
        Fluorescence movie (already F/F0 or R/R0).
    mask : (H, W) bool array
        Tissue mask.
    config : SourceDetectionConfig or None
        Configuration. Uses defaults if None.
    verbose : bool
        Print diagnostic info.
    
    Returns
    -------
    sources_dict : dict
        Dictionary with keys as method names, values as (M, 2) arrays of (row, col) coords.
    metrics_dict : dict
        Dictionary with metric values for each detected source.
    """
    if config is None:
        config = SourceDetectionConfig()
    
    T, H, W = Y.shape
    mask = mask.astype(bool)
    
    # Mask the data
    Y_masked = Y * mask[np.newaxis, :, :]
    
    sources_dict = {}
    metrics_dict = {}
    
    # =========================================================================
    # METHOD 1: TEMPORAL VARIANCE
    # =========================================================================
    if 'variance' in config.methods:
        var_map = np.var(Y_masked, axis=0)
        var_map[~mask] = 0
        
        sources_var = _find_local_maxima_2d(
            var_map, config.n_sources, 
            exclusion_radius=config.spatial_exclusion_radius,
            mask=mask
        )
        sources_dict['variance'] = sources_var
        metrics_dict['variance'] = var_map[sources_var[:, 0], sources_var[:, 1]]
        
        if verbose:
            print(f"Variance method: found {len(sources_var)} sources")
            print(f"  Variance range: {metrics_dict['variance'].min():.3f} - {metrics_dict['variance'].max():.3f}")
    
    # =========================================================================
    # METHOD 2: EARLY RESPONDERS
    # =========================================================================
    if 'early_response' in config.methods:
        baseline = np.mean(Y_masked[config.baseline_window[0]:config.baseline_window[1]], axis=0)
        baseline[baseline == 0] = 1
        
        early_mean = np.mean(Y_masked[config.early_window[0]:config.early_window[1]], axis=0)
        response_mag = (early_mean - baseline) / baseline
        response_mag[~mask] = -np.inf
        
        sources_early = _find_local_maxima_2d(
            response_mag, config.n_sources,
            exclusion_radius=config.spatial_exclusion_radius,
            mask=mask
        )
        sources_dict['early_response'] = sources_early
        metrics_dict['early_response'] = response_mag[sources_early[:, 0], sources_early[:, 1]]
        
        if verbose:
            print(f"Early response method: found {len(sources_early)} sources")
            print(f"  Response magnitude range: {metrics_dict['early_response'].min():.3f} - {metrics_dict['early_response'].max():.3f}")
    
    # =========================================================================
    # METHOD 3: PEAK DETECTION
    # =========================================================================
    if 'peak_detection' in config.methods:
        peak_height_map = np.zeros((H, W))
        
        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    continue
                trace = Y_masked[:, i, j]
                peaks, properties = find_peaks(
                    trace, 
                    prominence=config.peak_prominence, 
                    distance=config.peak_distance
                )
                if len(peaks) > 0:
                    peak_height_map[i, j] = np.max(properties['prominences'])
        
        peak_height_map[~mask] = 0
        sources_peak = _find_local_maxima_2d(
            peak_height_map, config.n_sources,
            exclusion_radius=config.spatial_exclusion_radius,
            mask=mask
        )
        sources_dict['peak_detection'] = sources_peak
        metrics_dict['peak_detection'] = peak_height_map[sources_peak[:, 0], sources_peak[:, 1]]
        
        if verbose:
            print(f"Peak detection method: found {len(sources_peak)} sources")
            print(f"  Peak height range: {metrics_dict['peak_detection'].min():.3f} - {metrics_dict['peak_detection'].max():.3f}")
    
    # =========================================================================
    # METHOD 4: SPATIAL GRADIENT (Laplacian)
    # =========================================================================
    if 'gradient' in config.methods:
        mean_intensity = np.mean(Y_masked, axis=0)
        laplacian = -laplace(mean_intensity)  # Negative for local maxima
        laplacian[~mask] = -np.inf
        
        sources_grad = _find_local_maxima_2d(
            laplacian, config.n_sources,
            exclusion_radius=config.spatial_exclusion_radius,
            mask=mask
        )
        sources_dict['gradient'] = sources_grad
        metrics_dict['gradient'] = laplacian[sources_grad[:, 0], sources_grad[:, 1]]
        
        if verbose:
            print(f"Gradient method: found {len(sources_grad)} sources")
            print(f"  Laplacian range: {metrics_dict['gradient'].min():.3f} - {metrics_dict['gradient'].max():.3f}")
    
    return sources_dict, metrics_dict


def _find_local_maxima_2d(
    metric_map: np.ndarray, 
    n_sources: int, 
    exclusion_radius: int = 5, 
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Find n_sources local maxima in metric_map with spatial exclusion.
    
    Returns (n, 2) array of (row, col) coordinates.
    """
    if mask is not None:
        metric_map = metric_map.copy()
        metric_map[~mask] = -np.inf
    
    # Find local maxima using maximum filter
    local_max = maximum_filter(metric_map, size=exclusion_radius) == metric_map
    local_max[metric_map == -np.inf] = False
    local_max[np.isnan(metric_map)] = False
    
    # Get coordinates and values
    coords = np.argwhere(local_max)
    values = metric_map[local_max]
    
    # Sort by value (descending)
    sorted_idx = np.argsort(values)[::-1]
    coords_sorted = coords[sorted_idx]
    
    return coords_sorted[:n_sources]


def combine_source_candidates(
    sources_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, np.ndarray],
    method: str = 'variance',
    max_sources: Optional[int] = None,
    exclusion_radius: int = 5
) -> np.ndarray:
    """
    Combine source candidates from multiple detection methods.
    
    Parameters
    ----------
    sources_dict : dict
        Output from identify_point_sources.
    metrics_dict : dict
        Metrics from identify_point_sources.
    method : str
        Combination method:
        - 'variance': Use only variance-based sources
        - 'union': Take union of all sources, remove duplicates
        - 'intersection': Keep sources found by multiple methods
        - 'weighted': Score by how many methods found each location
    max_sources : int or None
        Maximum number of sources to return.
    exclusion_radius : int
        Radius for considering two sources as "same".
    
    Returns
    -------
    sources : (M, 2) array of (row, col) coordinates
    """
    if method == 'variance' and 'variance' in sources_dict:
        sources = sources_dict['variance']
    elif method == 'union':
        # Stack all sources
        all_sources = np.vstack(list(sources_dict.values()))
        sources = _remove_duplicate_sources(all_sources, exclusion_radius)
    elif method == 'intersection':
        # Find sources that appear in multiple methods
        all_sources = np.vstack(list(sources_dict.values()))
        sources = _find_consensus_sources(all_sources, exclusion_radius, min_votes=2)
    elif method == 'weighted':
        # Score each source by number of methods that found it
        all_sources = np.vstack(list(sources_dict.values()))
        sources = _find_consensus_sources(all_sources, exclusion_radius, min_votes=1)
    else:
        # Default to first available method
        sources = list(sources_dict.values())[0]
    
    if max_sources is not None and len(sources) > max_sources:
        sources = sources[:max_sources]
    
    return sources


def _remove_duplicate_sources(sources: np.ndarray, radius: int) -> np.ndarray:
    """Remove duplicate sources within radius of each other."""
    keep = []
    for i, src in enumerate(sources):
        is_dup = False
        for j in keep:
            if np.linalg.norm(src - sources[j]) < radius:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return sources[keep]


def _find_consensus_sources(
    sources: np.ndarray, 
    radius: int, 
    min_votes: int = 2
) -> np.ndarray:
    """Find sources that appear in multiple methods (voting)."""
    n = len(sources)
    votes = np.zeros(n, dtype=int)
    
    # Count how many nearby sources each has
    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(sources[i] - sources[j]) < radius:
                votes[i] += 1
    
    # Keep sources with enough votes
    keep_idx = votes >= min_votes
    if not np.any(keep_idx):
        # Fall back to top sources by vote count
        top_idx = np.argsort(votes)[::-1][:max(1, len(sources) // 2)]
        return sources[top_idx]
    
    return _remove_duplicate_sources(sources[keep_idx], radius)


def visualize_point_sources(
    Y: np.ndarray, 
    mask: np.ndarray, 
    sources_dict: Dict[str, np.ndarray], 
    metrics_dict: Dict[str, np.ndarray],
    time_point: int = 0, 
    fps: float = 1/0.533,
    figsize: Tuple[int, int] = (18, 12)
) -> None:
    """
    Visualize identified point sources overlaid on the image.
    
    Creates two plots:
    1. Spatial view: Sources on single frame and temporal mean
    2. Temporal traces: Time courses at each source location
    """
    n_methods = len(sources_dict)
    
    fig, axes = plt.subplots(2, n_methods, figsize=figsize, constrained_layout=True)
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    mean_img = np.mean(Y, axis=0)
    frame_img = Y[time_point]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for col_idx, (method, sources) in enumerate(sources_dict.items()):
        # Top row: single frame
        ax1 = axes[0, col_idx]
        vmax = np.percentile(frame_img[mask], 99) if mask.sum() > 0 else frame_img.max()
        ax1.imshow(frame_img, cmap='gray', vmin=0, vmax=vmax)
        
        for i, (row, col) in enumerate(sources):
            circle = Circle((col, row), radius=3, color=colors[i % 10], 
                          fill=False, linewidth=2)
            ax1.add_patch(circle)
            ax1.plot(col, row, 'x', color=colors[i % 10], markersize=8, markeredgewidth=2)
        
        ax1.set_title(f'{method.replace("_", " ").title()}\n(Frame {time_point})')
        ax1.axis('off')
        
        # Bottom row: temporal mean
        ax2 = axes[1, col_idx]
        vmax = np.percentile(mean_img[mask], 99) if mask.sum() > 0 else mean_img.max()
        ax2.imshow(mean_img, cmap='gray', vmin=0, vmax=vmax)
        
        for i, (row, col) in enumerate(sources):
            circle = Circle((col, row), radius=3, color=colors[i % 10], 
                          fill=False, linewidth=2)
            ax2.add_patch(circle)
            ax2.plot(col, row, 'x', color=colors[i % 10], markersize=8, markeredgewidth=2)
        
        ax2.set_title('Temporal Mean')
        ax2.axis('off')
    
    plt.suptitle('Identified Point Source Candidates', fontsize=16, y=0.98)
    plt.show()
    
    # Time traces plot
    fig2, axes2 = plt.subplots(n_methods, 1, figsize=(12, 3*n_methods), constrained_layout=True)
    if n_methods == 1:
        axes2 = [axes2]
    
    t = np.arange(Y.shape[0]) / fps
    
    for ax_idx, (method, sources) in enumerate(sources_dict.items()):
        ax = axes2[ax_idx]
        
        for i, (row, col) in enumerate(sources):
            trace = Y[:, row, col]
            ax.plot(t, trace, label=f'S{i+1} ({row},{col})', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('R/R₀')
        ax.set_title(f'{method.replace("_", " ").title()} - Source Time Traces')
        ax.legend(ncol=3, fontsize=8, frameon=False)
        ax.grid(alpha=0.3)
    
    plt.show()


def visualize_combined_sources(
    Y: np.ndarray,
    mask: np.ndarray,
    sources: np.ndarray,
    fps: float = 1/0.533,
    time_point: int = 0,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """Visualize the final combined source locations."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    
    mean_img = np.mean(Y, axis=0)
    frame_img = Y[time_point]
    var_img = np.var(Y, axis=0)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    
    for ax, (img, title) in zip(axes, [(frame_img, f'Frame {time_point}'), 
                                        (mean_img, 'Temporal Mean'),
                                        (var_img, 'Temporal Variance')]):
        vmax = np.percentile(img[mask], 99) if mask.sum() > 0 else img.max()
        ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
        
        for i, (row, col) in enumerate(sources):
            circle = Circle((col, row), radius=4, color=colors[i], 
                          fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.text(col + 5, row, f'{i+1}', color=colors[i], fontsize=10, fontweight='bold')
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle(f'Final Point Source Locations (M={len(sources)})', fontsize=14)
    plt.show()
    
    # Time traces
    fig2, ax = plt.subplots(figsize=(12, 5))
    t = np.arange(Y.shape[0]) / fps
    
    for i, (row, col) in enumerate(sources):
        trace = Y[:, row, col]
        ax.plot(t, trace, label=f'Source {i+1}', alpha=0.8, linewidth=1.5, color=colors[i])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('R/R₀')
    ax.set_title('Source Time Traces')
    ax.legend(ncol=4, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2) SPATIAL BINNING (from diffusion_v2)
# =============================================================================

def bin2d(arr2d: np.ndarray, by_h: int = 2, by_w: int = 2, 
          mode: str = "mean") -> np.ndarray:
    """Bin a 2D array by integer factors."""
    H, W = arr2d.shape
    H2, W2 = (H // by_h) * by_h, (W // by_w) * by_w
    
    if H2 < by_h or W2 < by_w:
        raise ValueError(f"Array too small for binning")
    
    x = arr2d[:H2, :W2]
    x = x.reshape(H2 // by_h, by_h, W2 // by_w, by_w)
    
    if mode == "mean":
        return x.mean(axis=(1, 3))
    elif mode == "max":
        return x.max(axis=(1, 3))
    elif mode == "sum":
        return x.sum(axis=(1, 3))
    else:
        raise ValueError(f"mode must be 'mean', 'max', or 'sum'")


def bin_movie(Y: np.ndarray, by_h: int = 2, by_w: int = 2, 
              mode: str = "mean") -> np.ndarray:
    """Bin a (T, H, W) movie spatially."""
    T, H, W = Y.shape
    H2, W2 = (H // by_h) * by_h, (W // by_w) * by_w
    
    if H2 < by_h or W2 < by_w:
        raise ValueError(f"Movie too small for binning")
    
    x = Y[:, :H2, :W2]
    x = x.reshape(T, H2 // by_h, by_h, W2 // by_w, by_w)
    
    if mode == "mean":
        return x.mean(axis=(2, 4))
    elif mode == "max":
        return x.max(axis=(2, 4))
    elif mode == "sum":
        return x.sum(axis=(2, 4))
    else:
        raise ValueError(f"mode must be 'mean', 'max', or 'sum'")


# =============================================================================
# 3) GRAPH LAPLACIAN (from diffusion_v2)
# =============================================================================

def build_masked_laplacian(
    mask: np.ndarray, 
    connectivity: int = 4
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build a graph Laplacian on masked pixels with no-flux boundary conditions.
    
    Returns
    -------
    L : (N, N) csr_matrix
    pix_to_node : (H, W) int32 array, -1 for outside mask
    node_to_pix : (N, 2) int32 array
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    
    pix_to_node = np.full((H, W), -1, dtype=np.int32)
    node_indices = np.argwhere(mask)
    N = len(node_indices)
    
    if N == 0:
        raise ValueError("Mask has no True pixels")
    
    pix_to_node[mask] = np.arange(N, dtype=np.int32)
    node_to_pix = node_indices.astype(np.int32)
    
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), 
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        raise ValueError(f"connectivity must be 4 or 8")
    
    rows_list, cols_list = [], []
    
    for dr, dc in offsets:
        nr = node_to_pix[:, 0] + dr
        nc = node_to_pix[:, 1] + dc
        
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        valid_idx = np.where(valid)[0]
        neighbor_nodes = pix_to_node[nr[valid_idx], nc[valid_idx]]
        has_neighbor = neighbor_nodes >= 0
        
        i_nodes = valid_idx[has_neighbor]
        j_nodes = neighbor_nodes[has_neighbor]
        
        rows_list.append(i_nodes)
        cols_list.append(j_nodes)
    
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
    
    data_off = np.ones(len(rows), dtype=np.float64)
    A_off = sp.csr_matrix((data_off, (rows, cols)), shape=(N, N))
    degrees = np.asarray(A_off.sum(axis=1)).ravel()
    diag = sp.diags([-degrees], [0], shape=(N, N), format="csr")
    L = A_off + diag
    
    return L, pix_to_node, node_to_pix


# =============================================================================
# 4) POINT SOURCE SIMULATION
# =============================================================================

class PointSourceSimulator:
    """
    Simulator for reaction-diffusion PDE with point sources.
    
    Unlike DiffusionSimulator (which uses general spatial source maps Phi),
    this uses delta-function sources at specific pixel locations.
    
    Parameters
    ----------
    L : (N, N) sparse matrix
        Graph Laplacian.
    source_node_indices : (M,) array
        Node indices where sources are located.
    dt : float
        Time step in seconds.
    theta : float
        Time discretization (1.0=backward Euler, 0.5=Crank-Nicolson).
    """
    
    def __init__(self, L: sp.csr_matrix, source_node_indices: np.ndarray,
                 dt: float, theta: float = 1.0):
        self.L = L.tocsr()
        self.N = L.shape[0]
        self.dt = dt
        self.theta = theta
        self.source_nodes = np.asarray(source_node_indices, dtype=np.int32)
        self.M = len(self.source_nodes)
        
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0,1]")
        
        # Build Phi as a sparse (N, M) matrix with 1s at source locations
        self.Phi = self._build_point_source_basis()
        
        self._I = sp.eye(self.N, format="csr")
        self._cached_D = None
        self._cached_k = None
        self._solve_left = None
        self._M_right = None
    
    def _build_point_source_basis(self) -> sp.csr_matrix:
        """Build sparse Phi matrix for point sources."""
        rows = self.source_nodes
        cols = np.arange(self.M)
        data = np.ones(self.M)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.M))
    
    def _update_cache(self, D: float, k: float) -> None:
        """Update cached factorization if D or k changed."""
        if D == self._cached_D and k == self._cached_k:
            return
        
        A = D * self.L - k * self._I
        M_left = self._I - self.theta * self.dt * A
        
        if self.theta < 1.0:
            self._M_right = self._I + (1.0 - self.theta) * self.dt * A
        else:
            self._M_right = None
        
        try:
            self._solve_left = spla.factorized(M_left.tocsc())
        except Exception as e:
            logger.warning(f"LU factorization failed: {e}")
            self._solve_left = None
            self._M_left_backup = M_left
        
        self._cached_D = D
        self._cached_k = k
    
    def simulate(self, y0: np.ndarray, a_t: np.ndarray,
                 D: float, k: float, g: float = 1.0) -> np.ndarray:
        """
        Simulate the reaction-diffusion equation with point sources.
        
        Parameters
        ----------
        y0 : (N,) array
            Initial condition.
        a_t : (T, M) array
            Source time courses.
        D, k, g : float
            Model parameters.
        
        Returns
        -------
        I : (T+1, N) array
            Simulated intensity.
        """
        y0 = np.asarray(y0, dtype=np.float64).ravel()
        a_t = np.asarray(a_t, dtype=np.float64)
        
        if y0.size != self.N:
            raise ValueError(f"y0 has size {y0.size}, expected {self.N}")
        if a_t.shape[1] != self.M:
            raise ValueError(f"a_t has {a_t.shape[1]} sources, expected {self.M}")
        
        T = a_t.shape[0]
        self._update_cache(D, k)
        
        out = np.zeros((T + 1, self.N), dtype=np.float64)
        out[0] = y0
        
        for t in range(T):
            # Source term: point sources at specific nodes
            src = g * self.Phi @ a_t[t]  # (N,)
            
            if self._M_right is not None:
                rhs = self._M_right @ out[t] + self.dt * src
            else:
                rhs = out[t] + self.dt * src
            
            if self._solve_left is not None:
                out[t + 1] = self._solve_left(rhs)
            else:
                out[t + 1], _ = spla.cg(self._M_left_backup, rhs, x0=out[t], tol=1e-8)
        
        return out


# =============================================================================
# 5) FITTING WITH POINT SOURCES
# =============================================================================

def fit_point_source_model(
    Y: np.ndarray,
    mask: np.ndarray,
    source_locations: np.ndarray,
    dt: float = 0.533,
    D0: float = 1.0,
    k0: float = 0.5,
    g0: float = 1.0,
    fit_a_t: bool = True,
    a_t_init: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    loss_subsample: Optional[int] = 20000,
    theta: float = 1.0,
    max_iter: int = 200,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fit diffusion parameters with fixed point source locations.
    
    This is the key function that addresses the identifiability problem:
    by fixing source locations as delta functions, the diffusion coefficient D
    MUST explain observed spatial spreading.
    
    Parameters
    ----------
    Y : (T, H, W) array
        Observed movie (F/F0 or R/R0).
    mask : (H, W) bool array
        Tissue mask.
    source_locations : (M, 2) array
        Point source coordinates as (row, col).
    dt : float
        Time step in seconds.
    D0, k0, g0 : float
        Initial parameter guesses.
    fit_a_t : bool
        If True, jointly optimize source time courses.
        If False, use a_t_init or initialize from data.
    a_t_init : (T, M) array or None
        Initial time courses. If None, initialized from data at source locations.
    bounds : dict or None
        Parameter bounds.
    loss_subsample : int or None
        Subsample pixels for faster optimization.
    theta : float
        Time discretization.
    max_iter : int
        Maximum optimization iterations.
    verbose : bool
        Print progress.
    
    Returns
    -------
    dict with fitted parameters and diagnostics.
    """
    mask = np.asarray(mask, dtype=bool)
    source_locations = np.asarray(source_locations, dtype=np.int32)
    
    T, H, W = Y.shape
    M = len(source_locations)
    
    # Build Laplacian
    L, pix_to_node, node_to_pix = build_masked_laplacian(mask)
    N = L.shape[0]
    
    # Get source node indices
    source_node_indices = np.array([
        pix_to_node[r, c] for r, c in source_locations
    ], dtype=np.int32)
    
    # Check sources are in mask
    if np.any(source_node_indices < 0):
        bad_sources = np.where(source_node_indices < 0)[0]
        raise ValueError(f"Sources {bad_sources} are outside the mask")
    
    # Extract masked data
    Yv = Y[:, mask]  # (T, N)
    y0 = Yv[0].astype(np.float64)
    
    # Initialize time courses from data at source locations
    if a_t_init is None:
        a_t_init = np.zeros((T, M), dtype=np.float64)
        for m, (r, c) in enumerate(source_locations):
            a_t_init[:, m] = Y[:, r, c]
        # High-pass filter to get transients
        a_t_init = a_t_init - gaussian_filter1d(a_t_init, sigma=5, axis=0)
        a_t_init = np.maximum(a_t_init, 0)
    
    # Create simulator
    sim = PointSourceSimulator(L, source_node_indices, dt, theta=theta)
    
    # Subsample pixels for loss
    if loss_subsample is not None and loss_subsample < N:
        rng = np.random.default_rng(42)
        pixel_idx = rng.choice(N, size=loss_subsample, replace=False)
        Yv_sub = Yv[:, pixel_idx]
    else:
        pixel_idx = slice(None)
        Yv_sub = Yv
    
    # Default bounds
    if bounds is None:
        bounds = {'D': (0.01, 50), 'k': (0.01, 5), 'g': (0.01, 50)}
    
    n_eval = [0]
    best_loss = [np.inf]
    best_params = [None]
    
    if fit_a_t:
        # Joint optimization of D, k, g, and a_t
        # This is more expensive but more accurate
        return _fit_joint(
            Y, mask, source_locations, sim, Yv, y0, pixel_idx, Yv_sub,
            a_t_init, D0, k0, g0, bounds, max_iter, verbose
        )
    else:
        # Only optimize D, k, g with fixed a_t
        return _fit_transport_only(
            sim, Yv, y0, pixel_idx, Yv_sub, a_t_init,
            D0, k0, g0, bounds, max_iter, verbose
        )


def _fit_transport_only(
    sim: PointSourceSimulator,
    Yv: np.ndarray,
    y0: np.ndarray,
    pixel_idx,
    Yv_sub: np.ndarray,
    a_t: np.ndarray,
    D0: float, k0: float, g0: float,
    bounds: Dict,
    max_iter: int,
    verbose: bool
) -> Dict[str, Any]:
    """Fit D, k, g with fixed time courses a_t."""
    
    log_D0 = np.log(D0)
    log_k0 = np.log(k0)
    log_g0 = np.log(g0)
    
    log_bounds = Bounds(
        [np.log(bounds['D'][0]), np.log(bounds['k'][0]), np.log(bounds['g'][0])],
        [np.log(bounds['D'][1]), np.log(bounds['k'][1]), np.log(bounds['g'][1])]
    )
    
    n_eval = [0]
    
    def loss(log_theta):
        D = np.exp(log_theta[0])
        k = np.exp(log_theta[1])
        g = np.exp(log_theta[2])
        
        Ihat = sim.simulate(y0, a_t, D, k, g)
        resid = Ihat[1:, pixel_idx] - Yv_sub
        mse = np.mean(resid**2)
        
        n_eval[0] += 1
        if verbose and n_eval[0] % 20 == 0:
            logger.info(f"Eval {n_eval[0]}: D={D:.4f}, k={k:.4f}, g={g:.4f}, MSE={mse:.6f}")
        
        return mse
    
    theta0 = np.array([log_D0, log_k0, log_g0])
    
    result = minimize(
        loss, theta0,
        method='L-BFGS-B',
        bounds=log_bounds,
        options={'maxiter': max_iter}
    )
    
    D_hat = np.exp(result.x[0])
    k_hat = np.exp(result.x[1])
    g_hat = np.exp(result.x[2])
    
    Ihat_final = sim.simulate(y0, a_t, D_hat, k_hat, g_hat)
    final_mse = np.mean((Ihat_final[1:, pixel_idx] - Yv_sub)**2)
    
    # Compute R²
    ss_res = np.sum((Ihat_final[1:] - Yv)**2)
    ss_tot = np.sum((Yv - Yv.mean())**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)
    
    logger.info(f"Fitted: D={D_hat:.4f}, k={k_hat:.4f}, g={g_hat:.4f}, "
                f"MSE={final_mse:.6f}, R²={r_squared:.4f}")
    
    return {
        'D': D_hat,
        'k': k_hat,
        'g': g_hat,
        'a_t': a_t,
        'mse': final_mse,
        'r_squared': r_squared,
        'Ihat': Ihat_final,
        'opt': result
    }


def _fit_joint(
    Y: np.ndarray,
    mask: np.ndarray,
    source_locations: np.ndarray,
    sim: PointSourceSimulator,
    Yv: np.ndarray,
    y0: np.ndarray,
    pixel_idx,
    Yv_sub: np.ndarray,
    a_t_init: np.ndarray,
    D0: float, k0: float, g0: float,
    bounds: Dict,
    max_iter: int,
    verbose: bool
) -> Dict[str, Any]:
    """
    Joint optimization of D, k, g and a_t using alternating minimization.
    
    1. Fix a_t, optimize D, k, g
    2. Fix D, k, g, optimize a_t (analytically or via gradient descent)
    3. Repeat until convergence
    """
    T, M = a_t_init.shape
    a_t = a_t_init.copy()
    
    D_hat, k_hat, g_hat = D0, k0, g0
    
    prev_mse = np.inf
    
    for outer_iter in range(10):  # Alternating iterations
        # Step 1: Fix a_t, optimize D, k, g
        if verbose:
            logger.info(f"\n=== Outer iteration {outer_iter + 1} ===")
            logger.info("Optimizing D, k, g...")
        
        result_transport = _fit_transport_only(
            sim, Yv, y0, pixel_idx, Yv_sub, a_t,
            D_hat, k_hat, g_hat, bounds, max_iter // 2, verbose=False
        )
        D_hat = result_transport['D']
        k_hat = result_transport['k']
        g_hat = result_transport['g']
        
        # Step 2: Fix D, k, g, optimize a_t
        if verbose:
            logger.info("Optimizing source time courses...")
        
        a_t = _optimize_time_courses(
            sim, Yv, y0, D_hat, k_hat, g_hat, a_t, 
            n_iter=50, verbose=False
        )
        
        # Check convergence
        Ihat = sim.simulate(y0, a_t, D_hat, k_hat, g_hat)
        mse = np.mean((Ihat[1:, pixel_idx] - Yv_sub)**2)
        
        if verbose:
            logger.info(f"  D={D_hat:.4f}, k={k_hat:.4f}, g={g_hat:.4f}, MSE={mse:.6f}")
        
        if abs(prev_mse - mse) / (prev_mse + 1e-10) < 1e-4:
            if verbose:
                logger.info("Converged!")
            break
        prev_mse = mse
    
    # Final evaluation
    Ihat_final = sim.simulate(y0, a_t, D_hat, k_hat, g_hat)
    final_mse = np.mean((Ihat_final[1:] - Yv)**2)
    
    ss_res = np.sum((Ihat_final[1:] - Yv)**2)
    ss_tot = np.sum((Yv - Yv.mean())**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)
    
    logger.info(f"\nFinal: D={D_hat:.4f}, k={k_hat:.4f}, g={g_hat:.4f}, "
                f"MSE={final_mse:.6f}, R²={r_squared:.4f}")
    
    return {
        'D': D_hat,
        'k': k_hat,
        'g': g_hat,
        'a_t': a_t,
        'mse': final_mse,
        'r_squared': r_squared,
        'Ihat': Ihat_final,
        'opt': None
    }


def _optimize_time_courses(
    sim: PointSourceSimulator,
    Yv: np.ndarray,
    y0: np.ndarray,
    D: float, k: float, g: float,
    a_t: np.ndarray,
    n_iter: int = 50,
    verbose: bool = False
) -> np.ndarray:
    """
    Optimize time courses a_t with D, k, g fixed.
    
    Uses coordinate descent: for each source m at each time t,
    find optimal a_t[t, m] given all other values.
    """
    T, M = a_t.shape
    a_t = a_t.copy()
    
    sim._update_cache(D, k)
    
    for iteration in range(n_iter):
        # For each source
        for m in range(M):
            # Compute residual without source m
            a_t_no_m = a_t.copy()
            a_t_no_m[:, m] = 0
            Ihat_no_m = sim.simulate(y0, a_t_no_m, D, k, g)
            
            residual = Yv - Ihat_no_m[1:]  # (T, N)
            
            # Effect of source m at node j = source_nodes[m]
            # When a_t[t, m] = 1, it adds dt * g * delta(j) to the source term
            # The response decays and diffuses from there
            
            # Simple greedy approach: at each time t, find alpha that
            # minimizes ||residual[t] - alpha * response||^2
            # where response is the contribution from source m at time t
            
            source_node = sim.source_nodes[m]
            
            for t in range(T):
                # Approximate response: just the local effect
                # (Full impulse response would require convolution)
                response = np.zeros(sim.N)
                response[source_node] = sim.dt * g
                
                # Least squares for alpha
                if np.dot(response, response) > 1e-10:
                    alpha = np.dot(residual[t], response) / np.dot(response, response)
                    a_t[t, m] = max(0, alpha)
        
        # Apply temporal smoothing
        a_t = gaussian_filter1d(a_t, sigma=1.0, axis=0, mode='nearest')
        a_t = np.maximum(a_t, 0)
    
    return a_t


# =============================================================================
# 6) MAIN FITTING FUNCTION
# =============================================================================

def fit_diffusion_point_sources(
    Y: np.ndarray,
    mask: np.ndarray,
    dt: float = 0.533,
    n_sources: int = 8,
    bin_factors: Tuple[int, int] = (1, 1),
    source_detection_config: Optional[SourceDetectionConfig] = None,
    source_combine_method: str = 'variance',
    D0: float = 1.0,
    k0: float = 0.5,
    g0: float = 1.0,
    fit_a_t: bool = True,
    loss_subsample: Optional[int] = 20000,
    theta: float = 1.0,
    verbose: bool = True
) -> PointSourceResult:
    """
    Full pipeline: detect sources → fit point source diffusion model.
    
    Parameters
    ----------
    Y : (T, H, W) array
        Fluorescence movie.
    mask : (H, W) bool array
        Tissue mask.
    dt : float
        Time step in seconds.
    n_sources : int
        Number of sources to detect.
    bin_factors : tuple
        Spatial binning (by_h, by_w).
    source_detection_config : SourceDetectionConfig or None
        Configuration for source detection.
    source_combine_method : str
        How to combine sources from different methods.
    D0, k0, g0 : float
        Initial parameter guesses.
    fit_a_t : bool
        Whether to jointly fit time courses.
    loss_subsample : int or None
        Subsample for faster optimization.
    theta : float
        Time discretization.
    verbose : bool
        Print progress.
    
    Returns
    -------
    PointSourceResult
        Complete fitting result.
    """
    # Step 1: Optional binning
    by_h, by_w = bin_factors
    if by_h > 1 or by_w > 1:
        if verbose:
            logger.info(f"Binning movie {by_h}x{by_w}...")
        Y_bin = bin_movie(Y, by_h, by_w, mode="mean")
        mask_bin = bin2d(mask.astype(float), by_h, by_w, mode="max") > 0.5
    else:
        Y_bin = Y
        mask_bin = mask.astype(bool)
    
    T, H, W = Y_bin.shape
    
    # Step 2: Detect sources
    if verbose:
        logger.info("Detecting point sources...")
    
    if source_detection_config is None:
        source_detection_config = SourceDetectionConfig(n_sources=n_sources)
    
    sources_dict, metrics_dict = identify_point_sources(
        Y_bin, mask_bin, source_detection_config, verbose=verbose
    )
    
    # Combine sources
    source_locations = combine_source_candidates(
        sources_dict, metrics_dict, 
        method=source_combine_method,
        max_sources=n_sources,
        exclusion_radius=source_detection_config.spatial_exclusion_radius
    )
    
    M = len(source_locations)
    if verbose:
        logger.info(f"Using {M} point sources")
    
    # Step 3: Build Laplacian
    L, pix_to_node, node_to_pix = build_masked_laplacian(mask_bin)
    N = L.shape[0]
    
    # Get source node indices
    source_node_indices = np.array([
        pix_to_node[r, c] for r, c in source_locations
    ], dtype=np.int32)
    
    # Step 4: Fit model
    if verbose:
        logger.info("Fitting point source diffusion model...")
    
    result = fit_point_source_model(
        Y_bin, mask_bin, source_locations,
        dt=dt, D0=D0, k0=k0, g0=g0,
        fit_a_t=fit_a_t,
        loss_subsample=loss_subsample,
        theta=theta,
        verbose=verbose
    )
    
    # Compute residuals on masked pixels
    Yv = Y_bin[:, mask_bin]
    residuals_masked = Yv - result['Ihat'][1:]
    
    # Package result
    return PointSourceResult(
        D=result['D'],
        k=result['k'],
        g=result['g'],
        source_locations=source_locations,
        source_node_indices=source_node_indices,
        a_t=result['a_t'],
        n_sources=M,
        Y_used=Y_bin,
        mask_used=mask_bin,
        Ihat_masked=result['Ihat'][1:],
        mse=result['mse'],
        r_squared=result['r_squared'],
        dt=dt,
        bin_factors=bin_factors,
        L=L,
        pix_to_node=pix_to_node,
        node_to_pix=node_to_pix,
        opt_result=result.get('opt'),
        source_detection_method=source_combine_method
    )


# =============================================================================
# 7) VISUALIZATION
# =============================================================================

def visualize_fit_result(
    result: PointSourceResult,
    time_points: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Visualize fitting results with observed vs reconstructed comparison."""
    
    if time_points is None:
        T = result.Y_used.shape[0]
        time_points = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    
    n_times = len(time_points)
    fig, axes = plt.subplots(3, n_times, figsize=figsize, constrained_layout=True)
    
    mask = result.mask_used
    
    for col, t in enumerate(time_points):
        # Row 1: Observed
        obs = result.Y_used[t]
        vmax = np.percentile(obs[mask], 99)
        axes[0, col].imshow(obs, cmap='viridis', vmin=0, vmax=vmax)
        axes[0, col].set_title(f't = {t}')
        axes[0, col].axis('off')
        
        # Row 2: Reconstructed
        recon = np.zeros_like(obs)
        recon[mask] = result.Ihat_masked[t]
        axes[1, col].imshow(recon, cmap='viridis', vmin=0, vmax=vmax)
        axes[1, col].axis('off')
        
        # Row 3: Residual
        resid = np.zeros_like(obs)
        resid[mask] = result.Y_used[t, mask] - result.Ihat_masked[t]
        vlim = np.percentile(np.abs(resid[mask]), 95)
        axes[2, col].imshow(resid, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axes[2, col].axis('off')
    
    axes[0, 0].set_ylabel('Observed', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    axes[2, 0].set_ylabel('Residual', fontsize=12)
    
    plt.suptitle(f'Point Source Model Fit\n'
                 f'D={result.D:.3f} px²/s, k={result.k:.3f} s⁻¹, g={result.g:.3f}, '
                 f'R²={result.r_squared:.3f}', fontsize=14)
    plt.show()


def visualize_source_activity(
    result: PointSourceResult,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """Visualize source locations and time courses."""
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    # Left: Source locations on mean image
    mean_img = np.mean(result.Y_used, axis=0)
    vmax = np.percentile(mean_img[result.mask_used], 99)
    
    axes[0].imshow(mean_img, cmap='gray', vmin=0, vmax=vmax)
    
    colors = plt.cm.tab10(np.linspace(0, 1, result.n_sources))
    for i, (row, col) in enumerate(result.source_locations):
        circle = Circle((col, row), radius=4, color=colors[i], 
                      fill=False, linewidth=2)
        axes[0].add_patch(circle)
        axes[0].text(col + 5, row, f'{i+1}', color=colors[i], fontsize=10, fontweight='bold')
    
    axes[0].set_title('Source Locations')
    axes[0].axis('off')
    
    # Right: Time courses
    t = np.arange(result.a_t.shape[0]) * result.dt
    for i in range(result.n_sources):
        axes[1].plot(t, result.a_t[:, i], label=f'Source {i+1}', 
                    color=colors[i], linewidth=1.5)
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Source Activity')
    axes[1].set_title('Source Time Courses')
    axes[1].legend(ncol=2, frameon=False)
    axes[1].grid(alpha=0.3)
    
    plt.show()


def plot_diffusion_spread(
    result: PointSourceResult,
    source_idx: int = 0,
    time_after_pulse: List[float] = [0.5, 1.0, 2.0, 5.0],
    figsize: Tuple[int, int] = (14, 3)
) -> None:
    """
    Visualize how diffusion spreads from a point source.
    
    Simulates a single pulse at the source and shows spreading over time.
    """
    mask = result.mask_used
    H, W = mask.shape
    
    sim = PointSourceSimulator(
        result.L, 
        result.source_node_indices,
        result.dt,
        theta=1.0
    )
    
    # Create impulse time course
    T_sim = int(max(time_after_pulse) / result.dt) + 10
    a_t_impulse = np.zeros((T_sim, result.n_sources))
    a_t_impulse[0, source_idx] = 1.0  # Single pulse at t=0
    
    # Simulate
    y0 = np.zeros(result.L.shape[0])
    Ihat = sim.simulate(y0, a_t_impulse, result.D, result.k, result.g)
    
    # Plot at specified times
    n_times = len(time_after_pulse)
    fig, axes = plt.subplots(1, n_times, figsize=figsize, constrained_layout=True)
    
    source_loc = result.source_locations[source_idx]
    
    for i, t_sec in enumerate(time_after_pulse):
        t_idx = int(t_sec / result.dt)
        if t_idx >= Ihat.shape[0]:
            t_idx = Ihat.shape[0] - 1
        
        img = np.zeros((H, W))
        img[mask] = Ihat[t_idx]
        
        vmax = img[mask].max() if img[mask].max() > 0 else 1
        axes[i].imshow(img, cmap='hot', vmin=0, vmax=vmax)
        
        # Mark source
        axes[i].plot(source_loc[1], source_loc[0], 'c+', markersize=15, markeredgewidth=2)
        axes[i].set_title(f't = {t_sec:.1f}s')
        axes[i].axis('off')
    
    plt.suptitle(f'Diffusion from Source {source_idx+1}\n'
                 f'D = {result.D:.3f} px²/s, k = {result.k:.3f} s⁻¹', fontsize=12)
    plt.show()


# =============================================================================
# 8) SMOKE TEST
# =============================================================================

def run_smoke_test(verbose: bool = True) -> bool:
    """Run a quick test to verify the module works."""
    
    if verbose:
        print("Running diffusion_v3 smoke test...")
    
    # Create synthetic data
    np.random.seed(42)
    T, H, W = 50, 60, 80
    
    # Create mask
    mask = np.zeros((H, W), dtype=bool)
    mask[10:50, 10:70] = True
    
    # Create synthetic movie with two sources
    Y = np.ones((T, H, W)) * 0.5
    
    # Source 1: top-left
    src1_r, src1_c = 20, 25
    for t in range(T):
        if t < 10:
            Y[t, src1_r-2:src1_r+3, src1_c-2:src1_c+3] += 0.5 * np.exp(-t/5)
    
    # Source 2: bottom-right  
    src2_r, src2_c = 40, 55
    for t in range(T):
        if 20 < t < 30:
            Y[t, src2_r-2:src2_r+3, src2_c-2:src2_c+3] += 0.8 * np.exp(-(t-25)**2/10)
    
    # Add noise
    Y += 0.05 * np.random.randn(T, H, W)
    Y = Y * mask[np.newaxis, :, :]
    
    try:
        # Test source detection
        config = SourceDetectionConfig(n_sources=5)
        sources_dict, metrics_dict = identify_point_sources(Y, mask, config, verbose=verbose)
        
        # Combine sources
        source_locations = combine_source_candidates(
            sources_dict, metrics_dict, method='variance', max_sources=3
        )
        
        if verbose:
            print(f"Detected {len(source_locations)} sources")
        
        # Test fitting
        result = fit_point_source_model(
            Y, mask, source_locations,
            dt=0.5, D0=0.5, k0=0.2, g0=1.0,
            fit_a_t=False,  # Keep simple for smoke test
            max_iter=20,
            verbose=False
        )
        
        if verbose:
            print(f"Fitted: D={result['D']:.4f}, k={result['k']:.4f}, g={result['g']:.4f}")
            print(f"MSE={result['mse']:.6f}, R²={result['r_squared']:.4f}")
        
        print("Smoke test PASSED ✓")
        return True
        
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run_smoke_test(verbose=True)
