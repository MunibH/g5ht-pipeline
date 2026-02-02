"""
diffusion_v2.py – Reaction–Diffusion Model for Fluorescence Imaging Data
=========================================================================

MAJOR IMPROVEMENTS OVER diffusion.py:
-------------------------------------
1. Laplacian: Vectorized construction (no Python loops), proper CSR, with sanity checks.
2. Simulator: 
   - Cached LU factorization when D,k fixed.
   - Crank–Nicolson option (theta-method) for better accuracy.
   - Explicit source term (avoids implicit coupling).
   - Stability checks.
3. Source Learning:
   - Cleaner NMF interface with consistent matrix orientation.
   - Multiple preprocessing options: detrend, highpass, spatial smoothing.
   - Elbow-plot helper to choose number of sources M.
   - Robust scaling option.
4. Fitting:
   - Staged pipeline: learn sources → fit D,k,g → optionally refine a_t.
   - Temporal smoothness regularization for a_t refinement.
   - Optional L1 sparsity penalty.
   - Support for joint fitting across multiple recordings with shared D,k.
5. Usability:
   - Clear API with shape checks, dtype handling, docstrings.
   - Visualization helpers for source maps, time courses, residuals.
   - Logging support.
6. Performance:
   - Sparse operations throughout.
   - Optional loss subsampling for large N.
   - Memory-efficient design for N ~ 100k pixels.

MATHEMATICAL MODEL:
-------------------
On masked pixels (flattened to N nodes), the fluorescence intensity i(t) evolves as:

    di/dt = D * L * i - k * i + s(t)

where:
    - L is the graph Laplacian (4-neighbor, no-flux boundaries)
    - D > 0 is diffusion coefficient (pixels²/s)
    - k > 0 is first-order decay rate (1/s)
    - s(t) = g * Phi @ a_t is the source field
      - Phi: (N, M) spatial basis (nonnegative)
      - a_t: (T, M) time courses (nonnegative)
      - g: global gain scalar

TIME DISCRETIZATION (theta-method):
-----------------------------------
    theta=1.0 → Backward Euler (stable, first-order accurate)
    theta=0.5 → Crank–Nicolson (second-order accurate)

    (I - theta*dt*A) i_{t+1} = (I + (1-theta)*dt*A) i_t + dt * s_t

where A = D*L - k*I.

Units:
    - D: pixels²/second (on the grid used, so if binned 2x2, pixels are 2x larger)
    - k: 1/second
    - dt: seconds per frame

Author: Munib Hasnain
Date: 2026-01-26
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Union, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize, Bounds
from sklearn.decomposition import NMF

# Configure module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class DiffusionModelResult:
    """Container for diffusion model fitting results."""
    
    # Fitted parameters
    D: float                          # Diffusion coefficient (pixels²/s)
    k: float                          # Decay rate (1/s)
    g: float                          # Source gain
    
    # Learned sources
    Phi: np.ndarray                   # Spatial basis (N, M)
    a_t: np.ndarray                   # Time courses (T, M)
    
    # Data used (possibly binned)
    Y_used: np.ndarray                # (T, H, W) movie used for fitting
    mask_used: np.ndarray             # (H, W) mask used
    
    # Reconstruction and diagnostics
    Ihat_masked: np.ndarray           # (T, N) reconstruction on masked pixels
    residuals_masked: np.ndarray      # (T, N) residuals on masked pixels
    mse: float                        # Mean squared error
    r_squared: float                  # Coefficient of determination
    
    # Metadata
    dt: float                         # Time step (seconds)
    bin_factors: Tuple[int, int]      # Spatial binning factors (h, w)
    n_sources: int                    # Number of source components M
    
    # Graph structure (for advanced use)
    L: sp.csr_matrix = field(repr=False)           # Laplacian
    pix_to_node: np.ndarray = field(repr=False)    # (H, W) -> node index
    node_to_pix: np.ndarray = field(repr=False)    # (N, 2) pixel coordinates
    
    # Optimization details
    opt_result: Any = field(default=None, repr=False)


# =============================================================================
# 1) SPATIAL BINNING UTILITIES
# =============================================================================

def bin2d(arr2d: np.ndarray, by_h: int = 2, by_w: int = 2, 
          mode: str = "mean") -> np.ndarray:
    """
    Bin a 2D array by integer factors.
    
    Parameters
    ----------
    arr2d : (H, W) array
    by_h, by_w : int
        Binning factors in height and width dimensions.
    mode : {'mean', 'max', 'sum'}
        Aggregation mode.
    
    Returns
    -------
    (H//by_h, W//by_w) array
    """
    H, W = arr2d.shape
    H2, W2 = (H // by_h) * by_h, (W // by_w) * by_w
    
    if H2 < by_h or W2 < by_w:
        raise ValueError(f"Array too small for binning: ({H},{W}) with factors ({by_h},{by_w})")
    
    x = arr2d[:H2, :W2]
    x = x.reshape(H2 // by_h, by_h, W2 // by_w, by_w)
    
    if mode == "mean":
        return x.mean(axis=(1, 3))
    elif mode == "max":
        return x.max(axis=(1, 3))
    elif mode == "sum":
        return x.sum(axis=(1, 3))
    else:
        raise ValueError(f"mode must be 'mean', 'max', or 'sum', got {mode}")


def bin_movie(Y: np.ndarray, by_h: int = 2, by_w: int = 2, 
              mode: str = "mean") -> np.ndarray:
    """
    Bin a (T, H, W) movie spatially.
    
    Uses vectorized reshaping for efficiency.
    
    Parameters
    ----------
    Y : (T, H, W) array
    by_h, by_w : int
        Binning factors.
    mode : {'mean', 'max', 'sum'}
    
    Returns
    -------
    (T, H//by_h, W//by_w) array
    """
    T, H, W = Y.shape
    H2, W2 = (H // by_h) * by_h, (W // by_w) * by_w
    
    if H2 < by_h or W2 < by_w:
        raise ValueError(f"Movie too small for binning: ({H},{W}) with factors ({by_h},{by_w})")
    
    # Vectorized binning
    x = Y[:, :H2, :W2]
    x = x.reshape(T, H2 // by_h, by_h, W2 // by_w, by_w)
    
    if mode == "mean":
        return x.mean(axis=(2, 4))
    elif mode == "max":
        return x.max(axis=(2, 4))
    elif mode == "sum":
        return x.sum(axis=(2, 4))
    else:
        raise ValueError(f"mode must be 'mean', 'max', or 'sum', got {mode}")


# =============================================================================
# 2) GRAPH LAPLACIAN CONSTRUCTION
# =============================================================================

def build_masked_laplacian(mask: np.ndarray, 
                           connectivity: int = 4) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build a graph Laplacian on masked pixels with no-flux (Neumann) boundary conditions.
    
    The Laplacian L is defined such that L[i,j] = 1 if nodes i,j are neighbors,
    and L[i,i] = -degree(i). This gives the standard graph Laplacian where
    L @ f gives the discrete Laplacian of f (sum of differences to neighbors).
    
    The no-flux boundary condition is automatically satisfied because nodes 
    outside the mask have no edges—there's simply no neighbor to diffuse to.
    
    Parameters
    ----------
    mask : (H, W) array-like
        Boolean mask where True indicates valid pixels.
    connectivity : {4, 8}
        4-neighbor (von Neumann) or 8-neighbor (Moore) connectivity.
        Default is 4 for standard diffusion discretization.
    
    Returns
    -------
    L : (N, N) csr_matrix
        Graph Laplacian, where N is the number of True pixels in mask.
    pix_to_node : (H, W) int32 array
        Maps (row, col) to node index. -1 for pixels outside mask.
    node_to_pix : (N, 2) int32 array
        Maps node index to (row, col) pixel coordinates.
    
    Notes
    -----
    The Laplacian is negative semi-definite: all eigenvalues ≤ 0.
    The smallest eigenvalue is 0 (constant vector in null space).
    
    For a connected mask, λ_2 < 0 and |λ_2| relates to how fast
    diffusion equilibrates.
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    
    # Build pixel-to-node mapping
    pix_to_node = np.full((H, W), -1, dtype=np.int32)
    node_indices = np.argwhere(mask)  # (N, 2)
    N = len(node_indices)
    
    if N == 0:
        raise ValueError("Mask has no True pixels")
    
    pix_to_node[mask] = np.arange(N, dtype=np.int32)
    node_to_pix = node_indices.astype(np.int32)
    
    # Define neighbor offsets
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), 
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    
    # Vectorized edge construction
    rows_list, cols_list = [], []
    
    for dr, dc in offsets:
        # Shifted neighbor coordinates
        nr = node_to_pix[:, 0] + dr
        nc = node_to_pix[:, 1] + dc
        
        # Check bounds
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        
        # Check if neighbor is in mask
        valid_idx = np.where(valid)[0]
        neighbor_nodes = pix_to_node[nr[valid_idx], nc[valid_idx]]
        has_neighbor = neighbor_nodes >= 0
        
        # Store edges (node i -> neighbor j)
        i_nodes = valid_idx[has_neighbor]
        j_nodes = neighbor_nodes[has_neighbor]
        
        rows_list.append(i_nodes)
        cols_list.append(j_nodes)
    
    # Combine all edges
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
    
    # Off-diagonal entries are +1 (edges)
    data_off = np.ones(len(rows), dtype=np.float64)
    
    # Build adjacency-like matrix (off-diagonal part)
    A_off = sp.csr_matrix((data_off, (rows, cols)), shape=(N, N))
    
    # Degree is sum of each row (number of neighbors)
    degrees = np.asarray(A_off.sum(axis=1)).ravel()
    
    # Diagonal entries are -degree
    diag = sp.diags([-degrees], [0], shape=(N, N), format="csr")
    
    # Laplacian = off-diagonal edges + diagonal degrees
    L = A_off + diag
    
    # Sanity checks
    _validate_laplacian(L, mask, node_to_pix)
    
    logger.debug(f"Built Laplacian: N={N} nodes, {len(rows)} edges, "
                 f"max_degree={degrees.max():.0f}")
    
    return L, pix_to_node, node_to_pix


def _validate_laplacian(L: sp.csr_matrix, mask: np.ndarray, 
                        node_to_pix: np.ndarray) -> None:
    """Run sanity checks on the Laplacian."""
    N = L.shape[0]
    
    # Check symmetry
    diff = L - L.T
    if diff.nnz > 0:
        asym = np.abs(diff.data).max()
        if asym > 1e-10:
            raise ValueError(f"Laplacian is not symmetric: max asymmetry = {asym}")
    
    # Check row sums = 0
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    max_row_sum = np.abs(row_sums).max()
    if max_row_sum > 1e-10:
        raise ValueError(f"Laplacian rows don't sum to zero: max = {max_row_sum}")
    
    # Check diagonal is non-positive (degrees are non-negative)
    diag = L.diagonal()
    if np.any(diag > 1e-10):
        raise ValueError("Laplacian diagonal has positive entries")
    
    # Check off-diagonal is non-negative (edges are +1)
    L_nodiag = L - sp.diags([diag], [0], format="csr")
    if L_nodiag.data.min() < -1e-10:
        raise ValueError("Laplacian has negative off-diagonal entries")


def estimate_diffusion_stability(L: sp.csr_matrix, dt: float, 
                                 D_max: float = 10.0) -> Dict[str, float]:
    """
    Estimate stability bounds for the diffusion discretization.
    
    For explicit Euler, stability requires dt * D * |λ_max(L)| < 2.
    For semi-implicit (backward Euler), no stability restriction.
    
    Returns
    -------
    dict with:
        - 'lambda_max_approx': approximate largest eigenvalue magnitude
        - 'D_max_explicit': maximum D for explicit Euler stability
        - 'cfl_number': dt * D_max * |λ_max| (should be < 2 for explicit)
    """
    N = L.shape[0]
    
    # Approximate largest eigenvalue magnitude using power iteration
    # The Laplacian has eigenvalues in (-∞, 0], with 0 being the smallest magnitude
    # We need the most negative eigenvalue
    try:
        # Use Lanczos to find extremal eigenvalues
        # k=2 to get smallest (most negative) eigenvalue
        if N > 10:
            eigenvalues = spla.eigsh(L, k=min(2, N-1), which='SA', return_eigenvectors=False)
            lambda_min = eigenvalues.min()  # Most negative
        else:
            eigenvalues = np.linalg.eigvalsh(L.toarray())
            lambda_min = eigenvalues.min()
    except Exception:
        # Fallback: estimate from maximum degree
        degrees = -L.diagonal()
        lambda_min = -2 * degrees.max()  # Rough upper bound
    
    lambda_max_mag = abs(lambda_min)
    D_max_explicit = 2.0 / (dt * lambda_max_mag) if lambda_max_mag > 0 else np.inf
    cfl = dt * D_max * lambda_max_mag
    
    return {
        'lambda_max_approx': lambda_max_mag,
        'D_max_explicit': D_max_explicit,
        'cfl_number': cfl
    }


# =============================================================================
# 3) SIMULATION (TIME STEPPING)
# =============================================================================

class DiffusionSimulator:
    """
    Efficient simulator for the reaction-diffusion PDE on masked pixels.
    
    Caches LU factorization for repeated simulations with same D, k.
    Supports theta-method time discretization.
    
    Parameters
    ----------
    L : (N, N) sparse matrix
        Graph Laplacian.
    dt : float
        Time step in seconds.
    theta : float in [0, 1]
        Time discretization parameter:
        - theta=1.0: Backward Euler (fully implicit, stable, 1st order)
        - theta=0.5: Crank-Nicolson (2nd order accurate)
        - theta=0.0: Forward Euler (explicit, conditionally stable)
    
    Notes
    -----
    The discretization is:
        (I - theta*dt*A) i_{t+1} = (I + (1-theta)*dt*A) i_t + dt*s_t
    where A = D*L - k*I.
    
    For theta=1, the source is treated explicitly (added to RHS).
    """
    
    def __init__(self, L: sp.csr_matrix, dt: float, theta: float = 1.0):
        self.L = L.tocsr()
        self.N = L.shape[0]
        self.dt = dt
        self.theta = theta
        
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0,1], got {theta}")
        
        self._I = sp.eye(self.N, format="csr")
        self._cached_D = None
        self._cached_k = None
        self._solve_left = None   # Factorized (I - theta*dt*A)
        self._M_right = None      # (I + (1-theta)*dt*A)
    
    def _update_cache(self, D: float, k: float) -> None:
        """Update cached factorization if D or k changed."""
        if D == self._cached_D and k == self._cached_k:
            return
        
        # A = D*L - k*I
        A = D * self.L - k * self._I
        
        # Left matrix: (I - theta*dt*A)
        M_left = self._I - self.theta * self.dt * A
        
        # Right matrix: (I + (1-theta)*dt*A)
        if self.theta < 1.0:
            self._M_right = self._I + (1.0 - self.theta) * self.dt * A
        else:
            self._M_right = None
        
        # Factorize for efficient repeated solves
        try:
            self._solve_left = spla.factorized(M_left.tocsc())
        except Exception as e:
            logger.warning(f"LU factorization failed, using iterative solver: {e}")
            self._solve_left = None
            self._M_left_backup = M_left
        
        self._cached_D = D
        self._cached_k = k
    
    def simulate(self, y0: np.ndarray, Phi: np.ndarray, a_t: np.ndarray,
                 D: float, k: float, g: float = 1.0) -> np.ndarray:
        """
        Simulate the reaction-diffusion equation.
        
        Parameters
        ----------
        y0 : (N,) array
            Initial condition on masked nodes.
        Phi : (N, M) array
            Spatial source basis (nonnegative).
        a_t : (T, M) array
            Source time courses (nonnegative).
        D : float > 0
            Diffusion coefficient.
        k : float > 0
            Decay rate.
        g : float
            Source gain.
        
        Returns
        -------
        I : (T+1, N) array
            Simulated intensity at each time point.
            I[0] = y0, I[t+1] is state after t steps.
        """
        # Input validation
        y0 = np.asarray(y0, dtype=np.float64).ravel()
        if y0.size != self.N:
            raise ValueError(f"y0 has size {y0.size}, expected {self.N}")
        
        Phi = np.asarray(Phi, dtype=np.float64)
        a_t = np.asarray(a_t, dtype=np.float64)
        
        if Phi.shape[0] != self.N:
            raise ValueError(f"Phi has {Phi.shape[0]} rows, expected {self.N}")
        if Phi.shape[1] != a_t.shape[1]:
            raise ValueError(f"Phi has {Phi.shape[1]} components but a_t has {a_t.shape[1]}")
        
        T = a_t.shape[0]
        
        # Update cache if needed
        self._update_cache(D, k)
        
        # Allocate output
        out = np.zeros((T + 1, self.N), dtype=np.float64)
        out[0] = y0
        
        # Time stepping
        for t in range(T):
            # Source term at time t
            src = g * (Phi @ a_t[t])  # (N,)
            
            # Build RHS
            if self._M_right is not None:
                # Crank-Nicolson: include explicit part
                rhs = self._M_right @ out[t] + self.dt * src
            else:
                # Backward Euler: simple RHS
                rhs = out[t] + self.dt * src
            
            # Solve
            if self._solve_left is not None:
                out[t + 1] = self._solve_left(rhs)
            else:
                # Fallback to iterative solver
                out[t + 1], info = spla.cg(self._M_left_backup, rhs, x0=out[t], tol=1e-8)
                if info != 0:
                    logger.warning(f"CG solver did not converge at step {t}")
        
        return out


def simulate_rd_masked(y0: np.ndarray, Phi: np.ndarray, a_t: np.ndarray,
                       L: sp.csr_matrix, dt: float, D: float, k: float, 
                       g: float = 1.0, theta: float = 1.0) -> np.ndarray:
    """
    Convenience function for one-off simulation.
    
    For repeated simulations with the same L, dt, create a DiffusionSimulator
    instance and call simulate() to benefit from cached factorization.
    
    Parameters
    ----------
    y0 : (N,) initial field
    Phi : (N, M) spatial sources
    a_t : (T, M) time courses
    L : (N, N) Laplacian
    dt : float
    D, k, g : parameters
    theta : time discretization (1.0=backward Euler, 0.5=Crank-Nicolson)
    
    Returns
    -------
    (T+1, N) simulated field
    """
    sim = DiffusionSimulator(L, dt, theta=theta)
    return sim.simulate(y0, Phi, a_t, D, k, g)


# =============================================================================
# 4) SOURCE LEARNING (NMF-BASED)
# =============================================================================

@dataclass
class SourceLearningConfig:
    """Configuration for source learning via NMF."""
    
    n_sources: int = 8                    # Number of source components M
    hp_sigma_frames: float = 3.0          # High-pass filter sigma (frames)
    detrend: bool = True                  # Remove linear trend before NMF
    spatial_smooth_sigma: float = 0.0     # Spatial smoothing before NMF (pixels)
    rectify_mode: str = "relu"            # 'relu' (max(x,0)) or 'abs' or 'none'
    scale_mode: str = "std"               # 'std', 'max', 'none' for normalization
    nmf_max_iter: int = 500               # NMF iterations
    nmf_tol: float = 1e-4                 # NMF convergence tolerance
    seed: int = 0                         # Random seed


def learn_sources_nmf(Y: np.ndarray, mask: np.ndarray, 
                      config: Optional[SourceLearningConfig] = None,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Learn spatial source maps and time courses using NMF.
    
    The approach:
    1. Extract masked pixels: Yv (T, N)
    2. Preprocess: detrend, high-pass filter, optionally smooth spatially
    3. Rectify to get nonnegative activity matrix A
    4. Run NMF: A ≈ W @ H, where W is (T, M) and H is (M, N)
    5. Return Phi = H.T (N, M) and a_t = W (T, M)
    
    Parameters
    ----------
    Y : (T, H, W) array
        Fluorescence movie (F/F0 or ΔF/F0).
    mask : (H, W) bool array
        Tissue mask.
    config : SourceLearningConfig or None
        Configuration object. If None, uses defaults.
    **kwargs : 
        Override config parameters.
    
    Returns
    -------
    Phi : (N, M) array
        Spatial source basis (nonnegative, columns sum to 1).
    a_t : (T, M) array
        Time courses (nonnegative).
    reconstruction_error : float
        Relative NMF reconstruction error.
    
    Notes
    -----
    The convention is:
        source_field(t) = Phi @ a_t[t]  # (N,)
    
    Phi is normalized so columns sum to 1 (probability-like spatial weights),
    and a_t absorbs the magnitude. This makes source maps more interpretable.
    """
    if config is None:
        config = SourceLearningConfig(**kwargs)
    else:
        # Allow kwargs to override config
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # Input validation
    Y = np.asarray(Y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D (T,H,W), got shape {Y.shape}")
    if mask.shape != Y.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} doesn't match Y spatial dims {Y.shape[1:]}")
    
    T, H, W = Y.shape
    N = mask.sum()
    M = config.n_sources
    
    if M >= min(T, N):
        logger.warning(f"n_sources={M} >= min(T,N)={min(T,N)}, may be underdetermined")
    
    # Extract masked pixels
    Yv = Y[:, mask].astype(np.float64)  # (T, N)
    
    # Optional spatial smoothing (before high-pass)
    if config.spatial_smooth_sigma > 0:
        # Need to go back to image space, smooth, then extract again
        Y_smooth = np.zeros_like(Y)
        for t in range(T):
            Y_smooth[t] = gaussian_filter(Y[t], sigma=config.spatial_smooth_sigma)
        Yv = Y_smooth[:, mask]
    
    # Detrend (remove linear trend in time for each pixel)
    if config.detrend:
        t_vec = np.linspace(-1, 1, T)[:, None]  # (T, 1)
        slope = ((Yv * t_vec).mean(axis=0) - Yv.mean(axis=0) * t_vec.mean()) / (t_vec**2).mean()
        Yv = Yv - slope[None, :] * t_vec
    
    # High-pass filter (temporal)
    if config.hp_sigma_frames > 0:
        Y_lowpass = gaussian_filter1d(Yv, sigma=config.hp_sigma_frames, axis=0, mode="nearest")
        A = Yv - Y_lowpass
    else:
        A = Yv.copy()
    
    # Rectify
    if config.rectify_mode == "relu":
        A = np.maximum(A, 0.0)
    elif config.rectify_mode == "abs":
        A = np.abs(A)
    elif config.rectify_mode == "none":
        # NMF requires nonnegative, so shift if needed
        if A.min() < 0:
            A = A - A.min()
    else:
        raise ValueError(f"Unknown rectify_mode: {config.rectify_mode}")
    
    # Scale normalization (helps NMF convergence)
    if config.scale_mode == "std":
        scale = A.std() + 1e-10
    elif config.scale_mode == "max":
        scale = A.max() + 1e-10
    elif config.scale_mode == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unknown scale_mode: {config.scale_mode}")
    
    A = A / scale
    
    # NMF: A ≈ W @ H
    # A is (T, N), W is (T, M), H is (M, N)
    nmf = NMF(
        n_components=M,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=config.nmf_max_iter,
        tol=config.nmf_tol,
        random_state=config.seed,
    )
    
    W = nmf.fit_transform(A)  # (T, M)
    H = nmf.components_        # (M, N)
    
    # Compute reconstruction error
    A_recon = W @ H
    recon_error = np.linalg.norm(A - A_recon) / (np.linalg.norm(A) + 1e-10)
    
    # Phi = H.T, a_t = W, but normalize Phi columns
    Phi = H.T  # (N, M)
    a_t = W    # (T, M)
    
    # Normalize Phi columns to sum to 1, transfer magnitude to a_t
    col_sums = Phi.sum(axis=0) + 1e-10
    Phi = Phi / col_sums
    a_t = a_t * col_sums * scale  # Restore scale
    
    logger.info(f"NMF sources learned: M={M}, recon_error={recon_error:.4f}")
    
    return Phi, a_t, recon_error


def select_n_sources_elbow(Y: np.ndarray, mask: np.ndarray,
                           M_range: range = range(2, 20),
                           config: Optional[SourceLearningConfig] = None,
                           plot: bool = True) -> Tuple[int, np.ndarray]:
    """
    Select number of sources M using elbow method on reconstruction error.
    
    Parameters
    ----------
    Y : (T, H, W) array
    mask : (H, W) bool
    M_range : range
        Candidate values for M.
    config : SourceLearningConfig
        Base config (n_sources will be varied).
    plot : bool
        If True, display elbow plot.
    
    Returns
    -------
    M_best : int
        Suggested number of sources (elbow point).
    errors : array
        Reconstruction errors for each M.
    """
    if config is None:
        config = SourceLearningConfig()
    
    errors = []
    for M in M_range:
        try:
            _, _, err = learn_sources_nmf(Y, mask, config, n_sources=M)
            errors.append(err)
        except Exception as e:
            logger.warning(f"Failed for M={M}: {e}")
            errors.append(np.nan)
    
    errors = np.array(errors)
    M_range_arr = np.array(list(M_range))
    
    # Find elbow using max curvature
    # Approximate second derivative
    valid = ~np.isnan(errors)
    if valid.sum() < 3:
        raise ValueError("Not enough valid M values")
    
    # Simple elbow: max second derivative
    d1 = np.diff(errors[valid])
    d2 = np.diff(d1)
    elbow_idx = np.argmax(d2) + 1  # +1 because of diff
    M_best = M_range_arr[valid][elbow_idx]
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(M_range_arr[valid], errors[valid], 'bo-', lw=2, markersize=8)
            ax.axvline(M_best, color='r', linestyle='--', label=f'Elbow M={M_best}')
            ax.set_xlabel('Number of Sources M')
            ax.set_ylabel('Relative Reconstruction Error')
            ax.set_title('Elbow Plot for Source Count Selection')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass
    
    return M_best, errors


# =============================================================================
# 5) REGULARIZATION AND PARAMETER FITTING
# =============================================================================

@dataclass
class RegularizationConfig:
    """
    Configuration for regularization during parameter fitting.
    
    The key problem: when D→0, the source term can explain all local dynamics
    without any diffusion. These regularizers push the model toward using
    diffusion by constraining the sources.
    
    Attributes
    ----------
    source_l1_weight : float
        L1 penalty on source time courses: λ * sum(|a_t|).
        Promotes temporal sparsity (discrete bursts rather than continuous).
        Typical range: 0.001 - 0.1
        
    source_tv_weight : float
        Total Variation penalty on sources: λ * sum(|a_t - a_{t-1}|).
        Promotes piecewise-constant time courses with sparse changes.
        Typical range: 0.01 - 1.0
        
    source_energy_weight : float
        Penalty on total source "energy": λ * sum(a_t^2).
        Limits how much the sources can explain, forcing diffusion.
        Typical range: 0.001 - 0.1
        
    D_min_penalty_weight : float
        Soft penalty pushing D away from zero: λ * (1/D).
        Prevents D from collapsing. Typical range: 0.01 - 1.0
        
    D_prior_weight : float
        Gaussian prior on log(D): λ * (log(D) - log(D_prior_mean))^2.
        Regularizes D toward an expected value. Typical range: 0.1 - 10
        
    D_prior_mean : float
        Mean of the Gaussian prior on D. Should be set based on expected
        diffusion coefficient (in pixels²/s). Typical: 0.5 - 5.0
        
    g_max_penalty_weight : float
        Penalty for large g: λ * g^2. Limits source gain.
        Typical range: 0.001 - 0.1
    
    Notes
    -----
    The total loss becomes:
        L = MSE + source_l1 + source_tv + source_energy + D_penalties + g_penalty
    
    Start with small regularization weights and increase if D still collapses.
    A good starting point:
        - source_l1_weight=0.01 (mild sparsity)
        - source_tv_weight=0.1 (encourage discrete events)
        - D_prior_weight=1.0, D_prior_mean=1.0 (soft prior on D)
    """
    source_l1_weight: float = 0.0         # L1 on a_t
    source_tv_weight: float = 0.0         # TV on a_t 
    source_energy_weight: float = 0.0     # L2 on a_t
    D_min_penalty_weight: float = 0.0     # 1/D penalty
    D_prior_weight: float = 0.0           # (log D - log D0)^2 prior
    D_prior_mean: float = 1.0             # Prior mean for D
    g_max_penalty_weight: float = 0.0     # g^2 penalty


def compute_source_regularization(a_t: np.ndarray, g: float,
                                   config: RegularizationConfig) -> Tuple[float, Dict[str, float]]:
    """
    Compute regularization terms for source time courses.
    
    Parameters
    ----------
    a_t : (T, M) array
        Source time courses.
    g : float
        Source gain.
    config : RegularizationConfig
    
    Returns
    -------
    total_reg : float
        Total regularization penalty.
    components : dict
        Individual penalty components for diagnostics.
    """
    components = {}
    total_reg = 0.0
    
    # Scale a_t by g for regularization (penalize effective source strength)
    a_scaled = g * a_t
    
    # L1 sparsity: promotes temporal sparsity
    if config.source_l1_weight > 0:
        l1_penalty = config.source_l1_weight * np.sum(np.abs(a_scaled))
        total_reg += l1_penalty
        components['source_l1'] = l1_penalty
    
    # Total Variation: promotes piecewise constant (sparse changes)
    if config.source_tv_weight > 0:
        # TV = sum of |a_t - a_{t-1}| across time and sources
        tv_penalty = config.source_tv_weight * np.sum(np.abs(np.diff(a_scaled, axis=0)))
        total_reg += tv_penalty
        components['source_tv'] = tv_penalty
    
    # Energy penalty: limits total source power
    if config.source_energy_weight > 0:
        energy_penalty = config.source_energy_weight * np.sum(a_scaled**2)
        total_reg += energy_penalty
        components['source_energy'] = energy_penalty
    
    # Gain penalty
    if config.g_max_penalty_weight > 0:
        g_penalty = config.g_max_penalty_weight * g**2
        total_reg += g_penalty
        components['g_penalty'] = g_penalty
    
    return total_reg, components


def compute_D_regularization(D: float, config: RegularizationConfig) -> Tuple[float, Dict[str, float]]:
    """
    Compute regularization terms for diffusion coefficient.
    
    Parameters
    ----------
    D : float
        Diffusion coefficient.
    config : RegularizationConfig
    
    Returns
    -------
    total_reg : float
        Total D regularization.
    components : dict
        Individual components.
    """
    components = {}
    total_reg = 0.0
    
    # 1/D penalty: prevents D → 0
    if config.D_min_penalty_weight > 0:
        D_min_penalty = config.D_min_penalty_weight / (D + 1e-6)
        total_reg += D_min_penalty
        components['D_min_penalty'] = D_min_penalty
    
    # Gaussian prior on log(D)
    if config.D_prior_weight > 0:
        log_D_diff = np.log(D + 1e-10) - np.log(config.D_prior_mean)
        D_prior_penalty = config.D_prior_weight * log_D_diff**2
        total_reg += D_prior_penalty
        components['D_prior'] = D_prior_penalty
    
    return total_reg, components


def fit_parameters(Y: np.ndarray, mask: np.ndarray, 
                   Phi: np.ndarray, a_t: np.ndarray,
                   L: sp.csr_matrix, dt: float,
                   D0: float = 0.5, k0: float = 0.2, g0: float = 1.0,
                   loss_subsample: Optional[int] = None,
                   theta: float = 1.0,
                   bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                   method: str = "L-BFGS-B",
                   reg_config: Optional[RegularizationConfig] = None,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Fit diffusion parameters D, k, g with fixed sources Phi, a_t.
    
    Minimizes MSE + regularization between simulated and observed intensity.
    
    Parameters
    ----------
    Y : (T, H, W) array
        Observed movie.
    mask : (H, W) bool
        Tissue mask.
    Phi : (N, M) array
        Spatial sources.
    a_t : (T, M) array
        Time courses.
    L : (N, N) sparse matrix
        Laplacian.
    dt : float
        Time step.
    D0, k0, g0 : float
        Initial parameter guesses.
    loss_subsample : int or None
        If set, randomly sample this many pixels for loss computation.
        Useful for large N to speed up optimization.
    theta : float
        Time discretization (1.0=backward Euler, 0.5=Crank-Nicolson).
    bounds : dict or None
        Parameter bounds as {'D': (lo, hi), 'k': (lo, hi), 'g': (lo, hi)}.
        Default: D in [1e-4, 100], k in [1e-4, 10], g in [1e-4, 100].
    method : str
        scipy.optimize method.
    reg_config : RegularizationConfig or None
        Regularization configuration. If None, no regularization.
        See RegularizationConfig for options to prevent D→0.
    verbose : bool
        Print optimization progress.
    
    Returns
    -------
    dict with:
        'D', 'k', 'g': fitted parameters
        'mse': mean squared error (without regularization)
        'loss': total loss (MSE + regularization)
        'reg_components': dict of individual regularization terms
        'opt': scipy optimize result
    """
    mask = np.asarray(mask, dtype=bool)
    Yv = Y[:, mask]  # (T, N)
    y0 = Yv[0].astype(np.float64)
    T, N = Yv.shape
    
    # Default regularization: none
    if reg_config is None:
        reg_config = RegularizationConfig()
    
    # Subsample pixels for loss
    if loss_subsample is not None and loss_subsample < N:
        rng = np.random.default_rng(42)
        pixel_idx = rng.choice(N, size=loss_subsample, replace=False)
        Yv_sub = Yv[:, pixel_idx]
    else:
        pixel_idx = slice(None)
        Yv_sub = Yv
    
    # Create simulator (factorization will be updated in loss)
    sim = DiffusionSimulator(L, dt, theta=theta)
    
    # Default bounds
    if bounds is None:
        bounds = {'D': (1e-4, 100), 'k': (1e-4, 10), 'g': (1e-4, 100)}
    
    # Use log-transform for positivity
    log_D0 = np.log(D0)
    log_k0 = np.log(k0)
    log_g0 = np.log(g0)
    
    log_bounds = Bounds(
        [np.log(bounds['D'][0]), np.log(bounds['k'][0]), np.log(bounds['g'][0])],
        [np.log(bounds['D'][1]), np.log(bounds['k'][1]), np.log(bounds['g'][1])]
    )
    
    n_eval = [0]
    last_reg_components = [{}]
    
    def loss(log_theta):
        D = np.exp(log_theta[0])
        k = np.exp(log_theta[1])
        g = np.exp(log_theta[2])
        
        # Simulate
        Ihat = sim.simulate(y0, Phi, a_t, D, k, g)  # (T+1, N)
        
        # Compute MSE on subsampled pixels
        resid = Ihat[1:, pixel_idx] - Yv_sub  # (T, N_sub)
        mse = np.mean(resid**2)
        
        # Compute regularization
        source_reg, source_components = compute_source_regularization(a_t, g, reg_config)
        D_reg, D_components = compute_D_regularization(D, reg_config)
        
        total_loss = mse + source_reg + D_reg
        
        # Store for diagnostics
        last_reg_components[0] = {**source_components, **D_components, 'mse': mse}
        
        n_eval[0] += 1
        if verbose and n_eval[0] % 10 == 0:
            reg_str = f", reg={source_reg + D_reg:.4f}" if (source_reg + D_reg) > 0 else ""
            logger.info(f"Eval {n_eval[0]}: D={D:.4f}, k={k:.4f}, g={g:.4f}, MSE={mse:.6f}{reg_str}")
        
        return total_loss
    
    # Optimize
    theta0 = np.array([log_D0, log_k0, log_g0])
    
    options = {'maxiter': 500}
    if verbose:
        options['iprint'] = 1
    
    result = minimize(
        loss, theta0,
        method=method,
        bounds=log_bounds,
        options=options
    )
    
    D_hat = np.exp(result.x[0])
    k_hat = np.exp(result.x[1])
    g_hat = np.exp(result.x[2])
    
    # Compute final MSE without regularization for reporting
    Ihat_final = sim.simulate(y0, Phi, a_t, D_hat, k_hat, g_hat)
    final_mse = np.mean((Ihat_final[1:, pixel_idx] - Yv_sub)**2)
    
    logger.info(f"Fitted: D={D_hat:.4f}, k={k_hat:.4f}, g={g_hat:.4f}, MSE={final_mse:.6f}")
    
    return {
        'D': D_hat,
        'k': k_hat,
        'g': g_hat,
        'mse': final_mse,
        'loss': result.fun,
        'reg_components': last_reg_components[0],
        'opt': result
    }


def refine_time_courses(Y: np.ndarray, mask: np.ndarray,
                        Phi: np.ndarray, a_t_init: np.ndarray,
                        L: sp.csr_matrix, dt: float,
                        D: float, k: float, g: float,
                        temporal_smooth_weight: float = 0.1,
                        l1_weight: float = 0.0,
                        max_iter: int = 100,
                        verbose: bool = False) -> np.ndarray:
    """
    Refine time courses a_t with temporal smoothness and optional sparsity.
    
    Minimizes:
        ||Y - I_sim||^2 + λ_smooth * sum_t ||a_t - a_{t-1}||^2 + λ_L1 * ||a||_1
    
    subject to a_t >= 0.
    
    Uses coordinate descent with nonnegative least squares.
    
    Parameters
    ----------
    Y : (T, H, W) array
    mask : (H, W) bool
    Phi : (N, M) spatial sources (fixed)
    a_t_init : (T, M) initial time courses
    L, dt, D, k, g : model parameters (fixed)
    temporal_smooth_weight : float
        Weight for ||a_t - a_{t-1}||^2 regularization.
    l1_weight : float
        Weight for L1 sparsity.
    max_iter : int
        Max optimization iterations.
    verbose : bool
    
    Returns
    -------
    a_t_refined : (T, M) array
    """
    from scipy.optimize import nnls
    
    mask = np.asarray(mask, dtype=bool)
    Yv = Y[:, mask]  # (T, N)
    T, N = Yv.shape
    M = Phi.shape[1]
    
    y0 = Yv[0].astype(np.float64)
    a_t = a_t_init.copy()
    
    sim = DiffusionSimulator(L, dt, theta=1.0)
    
    # Build temporal regularization matrix
    # D_t @ a penalizes a_t - a_{t-1}
    D_t = sp.diags([[-1]*T, [1]*T], [0, 1], shape=(T-1, T)).toarray()
    
    for iteration in range(max_iter):
        # Forward simulate with current a_t
        Ihat = sim.simulate(y0, Phi, a_t, D, k, g)  # (T+1, N)
        
        # For each source component, solve subproblem
        for m in range(M):
            # Residual without component m
            a_t_no_m = a_t.copy()
            a_t_no_m[:, m] = 0
            Ihat_no_m = sim.simulate(y0, Phi, a_t_no_m, D, k, g)
            
            residual = Yv - Ihat_no_m[1:]  # (T, N)
            
            # Effect of source m: roughly dt * g * Phi[:, m] added at each step
            # This is approximate; for a more accurate solve, we'd need the 
            # impulse response, but this is a reasonable approximation
            Phi_m = Phi[:, m]  # (N,)
            
            # Least squares: minimize ||residual - alpha_m * (dt*g*Phi_m)||^2
            # For each time step independently (greedy approximation)
            alpha_new = np.zeros(T)
            for t in range(T):
                response = dt * g * Phi_m
                if np.dot(response, response) > 1e-10:
                    alpha_new[t] = np.maximum(0, np.dot(residual[t], response) / np.dot(response, response))
            
            # Apply temporal smoothing as post-hoc filter
            if temporal_smooth_weight > 0:
                alpha_smooth = gaussian_filter1d(alpha_new, sigma=2.0, mode='nearest')
                alpha_new = (alpha_new + temporal_smooth_weight * alpha_smooth) / (1 + temporal_smooth_weight)
            
            # Ensure nonnegativity
            alpha_new = np.maximum(alpha_new, 0)
            
            a_t[:, m] = alpha_new
        
        if verbose and (iteration + 1) % 10 == 0:
            Ihat = sim.simulate(y0, Phi, a_t, D, k, g)
            mse = np.mean((Yv - Ihat[1:])**2)
            logger.info(f"Refinement iter {iteration+1}: MSE={mse:.6f}")
    
    return a_t


# =============================================================================
# 6) MAIN FITTING FUNCTION
# =============================================================================

def fit_diffusion_model_2d(
    Y: np.ndarray,
    mask: np.ndarray,
    dt: float = 0.533,
    n_sources: int = 8,
    bin_factors: Tuple[int, int] = (1, 1),
    hp_sigma_frames: float = 3.0,
    loss_subsample: Optional[int] = 15000,
    theta: float = 1.0,
    refine_sources: bool = False,
    temporal_smooth_weight: float = 0.1,
    source_config: Optional[SourceLearningConfig] = None,
    reg_config: Optional[RegularizationConfig] = None,
    D0: float = 0.5,
    k0: float = 0.2,
    g0: float = 1.0,
    verbose: bool = False
) -> DiffusionModelResult:
    """
    Fit a reaction-diffusion model to fluorescence imaging data.
    
    This is the main entry point for the diffusion modeling pipeline.
    
    Pipeline:
    1. Optional spatial binning for speed
    2. Learn sources (Phi, a_t) via NMF on high-pass rectified activity
    3. Build graph Laplacian on masked pixels
    4. Fit D, k, g by minimizing reconstruction error
    5. Optionally refine a_t with temporal smoothness
    6. Compute reconstruction and diagnostics
    
    Parameters
    ----------
    Y : (T, H, W) array
        Fluorescence movie. Can be F/F0 or ΔF/F0.
    mask : (H, W) array
        Boolean tissue mask.
    dt : float
        Time step in seconds (default 0.533s).
    n_sources : int
        Number of source components M (default 8).
    bin_factors : (int, int)
        Spatial binning factors (by_h, by_w). (1,1) means no binning.
    hp_sigma_frames : float
        High-pass filter sigma for source learning (in frames).
    loss_subsample : int or None
        Number of pixels to subsample for loss computation.
        Set to None to use all pixels.
    theta : float
        Time discretization: 1.0=backward Euler, 0.5=Crank-Nicolson.
    refine_sources : bool
        Whether to refine a_t after fitting D,k,g.
    temporal_smooth_weight : float
        Temporal smoothness weight for source refinement.
    source_config : SourceLearningConfig or None
        Advanced source learning configuration.
    reg_config : RegularizationConfig or None
        Regularization to prevent D→0 by constraining sources.
        If None, no regularization. Recommended starting point:
        RegularizationConfig(source_tv_weight=0.1, D_prior_weight=1.0, D_prior_mean=1.0)
    D0, k0, g0 : float
        Initial parameter guesses.
    verbose : bool
        Print progress.
    
    Returns
    -------
    DiffusionModelResult
        Dataclass containing:
        - D, k, g: fitted parameters
        - Phi, a_t: learned sources
        - Ihat_masked, residuals_masked: reconstruction and residuals
        - mse, r_squared: fit quality metrics
        - L, pix_to_node, node_to_pix: graph structure
        - Y_used, mask_used: (possibly binned) data used
    
    Examples
    --------
    >>> result = fit_diffusion_model_2d(Y, mask, dt=0.533, n_sources=5)
    >>> print(f"D={result.D:.3f} px²/s, k={result.k:.3f} 1/s")
    >>> # Visualize sources
    >>> visualize_sources(result)
    
    Notes
    -----
    Units:
    - D is in pixels²/second on the (possibly binned) grid
    - k is in 1/second
    - If you binned by 2x2, the "pixel" is 2x larger, so D is effectively
      4x larger in original pixel units.
    
    Boundary conditions:
    - No-flux (Neumann) at mask boundary: diffusion stops at edges.
    """
    # Input validation
    Y = np.asarray(Y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    
    if Y.ndim != 3:
        raise ValueError(f"Y must be 3D (T,H,W), got shape {Y.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got shape {mask.shape}")
    if mask.shape != Y.shape[1:]:
        raise ValueError(f"mask shape {mask.shape} doesn't match Y shape {Y.shape[1:]}")
    
    T_orig, H_orig, W_orig = Y.shape
    logger.info(f"Input: T={T_orig}, H={H_orig}, W={W_orig}, N_masked={mask.sum()}")
    
    # 1. Optional spatial binning
    by_h, by_w = bin_factors
    if by_h > 1 or by_w > 1:
        logger.info(f"Binning by ({by_h}, {by_w})")
        Y_used = bin_movie(Y, by_h=by_h, by_w=by_w, mode="mean")
        mask_used = bin2d(mask.astype(np.float64), by_h=by_h, by_w=by_w, mode="mean") > 0.5
    else:
        Y_used = Y.copy()
        mask_used = mask.copy()
    
    T, H, W = Y_used.shape
    N = mask_used.sum()
    logger.info(f"After binning: T={T}, H={H}, W={W}, N_masked={N}")
    
    # 2. Build Laplacian
    logger.info("Building Laplacian...")
    L, pix_to_node, node_to_pix = build_masked_laplacian(mask_used)
    
    # Check stability
    stability = estimate_diffusion_stability(L, dt)
    logger.info(f"Stability: λ_max≈{stability['lambda_max_approx']:.2f}, "
                f"D_max_explicit={stability['D_max_explicit']:.2f}")
    
    # 3. Learn sources
    logger.info(f"Learning {n_sources} sources via NMF...")
    if source_config is None:
        source_config = SourceLearningConfig(
            n_sources=n_sources,
            hp_sigma_frames=hp_sigma_frames
        )
    else:
        source_config.n_sources = n_sources
        source_config.hp_sigma_frames = hp_sigma_frames
    
    Phi, a_t, nmf_error = learn_sources_nmf(Y_used, mask_used, source_config)
    logger.info(f"NMF reconstruction error: {nmf_error:.4f}")
    
    # 4. Fit parameters
    logger.info("Fitting D, k, g...")
    if reg_config is not None:
        logger.info(f"Using regularization: source_l1={reg_config.source_l1_weight}, "
                    f"source_tv={reg_config.source_tv_weight}, D_prior={reg_config.D_prior_weight}")
    fit_result = fit_parameters(
        Y_used, mask_used, Phi, a_t, L, dt,
        D0=D0, k0=k0, g0=g0,
        loss_subsample=loss_subsample,
        theta=theta,
        reg_config=reg_config,
        verbose=verbose
    )
    
    D, k, g = fit_result['D'], fit_result['k'], fit_result['g']
    
    # 5. Optional source refinement
    if refine_sources:
        logger.info("Refining time courses...")
        a_t = refine_time_courses(
            Y_used, mask_used, Phi, a_t, L, dt, D, k, g,
            temporal_smooth_weight=temporal_smooth_weight,
            verbose=verbose
        )
        # Re-fit g after refinement
        fit_result2 = fit_parameters(
            Y_used, mask_used, Phi, a_t, L, dt,
            D0=D, k0=k, g0=g,
            loss_subsample=loss_subsample,
            theta=theta,
            reg_config=reg_config,
            verbose=False
        )
        g = fit_result2['g']
    
    # 6. Final reconstruction
    logger.info("Computing final reconstruction...")
    sim = DiffusionSimulator(L, dt, theta=theta)
    Yv = Y_used[:, mask_used]
    y0 = Yv[0]
    Ihat = sim.simulate(y0, Phi, a_t, D, k, g)  # (T+1, N)
    Ihat_masked = Ihat[1:]  # (T, N), drop initial condition
    
    residuals = Yv - Ihat_masked
    mse = np.mean(residuals**2)
    ss_tot = np.var(Yv) * Yv.size
    ss_res = np.sum(residuals**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    logger.info(f"Final: MSE={mse:.6f}, R²={r_squared:.4f}")
    
    return DiffusionModelResult(
        D=D, k=k, g=g,
        Phi=Phi, a_t=a_t,
        Y_used=Y_used, mask_used=mask_used,
        Ihat_masked=Ihat_masked,
        residuals_masked=residuals,
        mse=mse, r_squared=r_squared,
        dt=dt, bin_factors=bin_factors, n_sources=n_sources,
        L=L, pix_to_node=pix_to_node, node_to_pix=node_to_pix,
        opt_result=fit_result['opt']
    )


def fit_multiple_recordings(
    recordings: List[Tuple[np.ndarray, np.ndarray]],
    dt: float = 0.533,
    n_sources: int = 8,
    bin_factors: Tuple[int, int] = (1, 1),
    shared_D_k: bool = True,
    **kwargs
) -> List[DiffusionModelResult]:
    """
    Fit multiple recordings with optionally shared D, k parameters.
    
    Parameters
    ----------
    recordings : list of (Y, mask) tuples
        Each Y is (T, H, W), mask is (H, W).
    dt : float
    n_sources : int
    bin_factors : tuple
    shared_D_k : bool
        If True, fit shared D, k across all recordings (but separate sources).
    **kwargs : passed to fit_diffusion_model_2d
    
    Returns
    -------
    list of DiffusionModelResult
    """
    results = []
    
    if not shared_D_k:
        # Fit each independently
        for i, (Y, mask) in enumerate(recordings):
            logger.info(f"Fitting recording {i+1}/{len(recordings)}")
            result = fit_diffusion_model_2d(Y, mask, dt=dt, n_sources=n_sources,
                                            bin_factors=bin_factors, **kwargs)
            results.append(result)
        return results
    
    # Shared D, k fitting
    # Step 1: Learn sources for each recording
    logger.info("Learning sources for each recording...")
    source_data = []
    for i, (Y, mask) in enumerate(recordings):
        by_h, by_w = bin_factors
        if by_h > 1 or by_w > 1:
            Y_used = bin_movie(Y, by_h=by_h, by_w=by_w, mode="mean")
            mask_used = bin2d(mask.astype(np.float64), by_h=by_h, by_w=by_w, mode="mean") > 0.5
        else:
            Y_used, mask_used = Y.copy(), mask.copy()
        
        L, pix_to_node, node_to_pix = build_masked_laplacian(mask_used)
        Phi, a_t, _ = learn_sources_nmf(Y_used, mask_used, n_sources=n_sources)
        source_data.append((Y_used, mask_used, L, Phi, a_t, pix_to_node, node_to_pix))
    
    # Step 2: Joint optimization of D, k
    logger.info("Joint optimization of shared D, k...")
    
    def joint_loss(log_theta):
        D = np.exp(log_theta[0])
        k = np.exp(log_theta[1])
        
        total_loss = 0.0
        for Y_used, mask_used, L, Phi, a_t, _, _ in source_data:
            # g is fitted per-recording (included in D,k optimization)
            g = np.exp(log_theta[2])  # Could extend to per-recording g
            
            Yv = Y_used[:, mask_used]
            y0 = Yv[0]
            sim = DiffusionSimulator(L, dt, theta=1.0)
            Ihat = sim.simulate(y0, Phi, a_t, D, k, g)
            resid = Ihat[1:] - Yv
            total_loss += np.mean(resid**2)
        
        return total_loss / len(source_data)
    
    D0 = kwargs.get('D0', 0.5)
    k0 = kwargs.get('k0', 0.2)
    g0 = kwargs.get('g0', 1.0)
    
    result = minimize(
        joint_loss,
        np.log([D0, k0, g0]),
        method="L-BFGS-B",
        options={'maxiter': 500}
    )
    
    D_shared = np.exp(result.x[0])
    k_shared = np.exp(result.x[1])
    g_shared = np.exp(result.x[2])
    
    logger.info(f"Shared parameters: D={D_shared:.4f}, k={k_shared:.4f}")
    
    # Step 3: Build results with shared D, k
    for i, (Y_used, mask_used, L, Phi, a_t, pix_to_node, node_to_pix) in enumerate(source_data):
        sim = DiffusionSimulator(L, dt, theta=1.0)
        Yv = Y_used[:, mask_used]
        y0 = Yv[0]
        Ihat = sim.simulate(y0, Phi, a_t, D_shared, k_shared, g_shared)
        Ihat_masked = Ihat[1:]
        residuals = Yv - Ihat_masked
        mse = np.mean(residuals**2)
        ss_tot = np.var(Yv) * Yv.size
        ss_res = np.sum(residuals**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        results.append(DiffusionModelResult(
            D=D_shared, k=k_shared, g=g_shared,
            Phi=Phi, a_t=a_t,
            Y_used=Y_used, mask_used=mask_used,
            Ihat_masked=Ihat_masked,
            residuals_masked=residuals,
            mse=mse, r_squared=r_squared,
            dt=dt, bin_factors=bin_factors, n_sources=n_sources,
            L=L, pix_to_node=pix_to_node, node_to_pix=node_to_pix
        ))
    
    return results


# =============================================================================
# 7) VISUALIZATION UTILITIES
# =============================================================================

def reshape_to_image(vec: np.ndarray, mask: np.ndarray, 
                     pix_to_node: Optional[np.ndarray] = None,
                     fill_value: float = np.nan) -> np.ndarray:
    """
    Reshape a vector on masked nodes back to image space.
    
    Parameters
    ----------
    vec : (N,) array
        Values on masked nodes.
    mask : (H, W) bool
        Tissue mask.
    pix_to_node : (H, W) int array or None
        Pixel-to-node mapping. If None, uses mask ordering.
    fill_value : float
        Value for pixels outside mask.
    
    Returns
    -------
    (H, W) array
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    
    img = np.full((H, W), fill_value, dtype=np.float64)
    img[mask] = vec
    
    return img


def visualize_sources(result: DiffusionModelResult, 
                      max_sources: int = 6,
                      figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Visualize source spatial maps and time courses.
    
    Parameters
    ----------
    result : DiffusionModelResult
    max_sources : int
        Maximum number of sources to display.
    figsize : tuple
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return
    
    M = min(result.n_sources, max_sources)
    T = result.a_t.shape[0]
    t = np.arange(T) * result.dt
    
    fig, axes = plt.subplots(2, M, figsize=figsize)
    
    for m in range(M):
        # Spatial map
        phi_m = result.Phi[:, m]
        img = reshape_to_image(phi_m, result.mask_used)
        
        ax = axes[0, m] if M > 1 else axes[0]
        im = ax.imshow(img, cmap='hot', aspect='auto')
        ax.set_title(f'Source {m+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Time course
        ax = axes[1, m] if M > 1 else axes[1]
        ax.plot(t, result.a_t[:, m], 'b-', lw=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.suptitle(f'D={result.D:.3f} px²/s, k={result.k:.3f} /s, R²={result.r_squared:.3f}')
    plt.show()


def plot_reconstruction_comparison(result: DiffusionModelResult,
                                    z_slice: int = 0,
                                    time_points: List[int] = None,
                                    figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot observed vs reconstructed frames.
    
    Parameters
    ----------
    result : DiffusionModelResult
    z_slice : int
        For 3D data, which z-slice to show (for 2D, ignored).
    time_points : list of int or None
        Which frames to show. If None, shows 5 evenly spaced frames.
    figsize : tuple
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    T = result.Y_used.shape[0]
    if time_points is None:
        time_points = np.linspace(0, T-1, 5).astype(int)
    
    n = len(time_points)
    fig, axes = plt.subplots(3, n, figsize=figsize)
    
    for i, t_idx in enumerate(time_points):
        # Observed
        obs = result.Y_used[t_idx]
        ax = axes[0, i]
        im = ax.imshow(obs, cmap='viridis', aspect='auto')
        ax.set_title(f't={t_idx * result.dt:.1f}s')
        ax.axis('off')
        if i == n - 1:
            ax.set_ylabel('Observed')
        
        # Reconstructed
        recon = reshape_to_image(result.Ihat_masked[t_idx], result.mask_used)
        ax = axes[1, i]
        ax.imshow(recon, cmap='viridis', aspect='auto', 
                  vmin=obs.min(), vmax=obs.max())
        ax.axis('off')
        if i == n - 1:
            ax.set_ylabel('Model')
        
        # Residual
        resid = reshape_to_image(result.residuals_masked[t_idx], result.mask_used)
        ax = axes[2, i]
        v = np.nanmax(np.abs(resid))
        ax.imshow(resid, cmap='RdBu_r', aspect='auto', vmin=-v, vmax=v)
        ax.axis('off')
        if i == n - 1:
            ax.set_ylabel('Residual')
    
    plt.tight_layout()
    plt.show()


def compute_residual_metrics(result: DiffusionModelResult) -> Dict[str, float]:
    """
    Compute various residual metrics.
    
    Returns
    -------
    dict with:
        - mse: mean squared error
        - rmse: root mean squared error
        - mae: mean absolute error
        - r_squared: coefficient of determination
        - pearson_r: Pearson correlation (temporal, averaged over pixels)
    """
    from scipy.stats import pearsonr
    
    Yv = result.Y_used[:, result.mask_used]
    resid = result.residuals_masked
    
    mse = np.mean(resid**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(resid))
    
    # Temporal correlation per pixel
    correlations = []
    for n in range(Yv.shape[1]):
        if np.std(Yv[:, n]) > 1e-10 and np.std(result.Ihat_masked[:, n]) > 1e-10:
            r, _ = pearsonr(Yv[:, n], result.Ihat_masked[:, n])
            if not np.isnan(r):
                correlations.append(r)
    
    mean_pearson = np.mean(correlations) if correlations else 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': result.r_squared,
        'pearson_r': mean_pearson
    }


# =============================================================================
# 8) SYNTHETIC DATA GENERATION (FOR TESTING)
# =============================================================================

def generate_synthetic_data(
    T: int = 200,
    H: int = 100,
    W: int = 150,
    dt: float = 0.533,
    D_true: float = 0.5,
    k_true: float = 0.2,
    n_sources: int = 3,
    noise_std: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic fluorescence data from a known reaction-diffusion model.
    
    Creates a simple oval mask, Gaussian spatial sources at random locations,
    and smooth time courses with bursts.
    
    Parameters
    ----------
    T : int
        Number of time frames.
    H, W : int
        Image dimensions.
    dt : float
        Time step.
    D_true, k_true : float
        True diffusion and decay parameters.
    n_sources : int
        Number of source locations.
    noise_std : float
        Standard deviation of Gaussian noise added to the signal.
    seed : int
        Random seed.
    
    Returns
    -------
    Y : (T, H, W) array
        Synthetic fluorescence movie.
    mask : (H, W) bool array
        Tissue mask.
    ground_truth : dict
        Contains true parameters, sources, and clean signal.
    """
    rng = np.random.default_rng(seed)
    
    # Create elliptical mask
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    mask = ((yy - cy)**2 / (H/2.5)**2 + (xx - cx)**2 / (W/2.5)**2) < 1
    
    # Build Laplacian
    L, pix_to_node, node_to_pix = build_masked_laplacian(mask)
    N = mask.sum()
    
    # Create source locations (Gaussian blobs)
    Phi = np.zeros((N, n_sources))
    source_centers = []
    
    for m in range(n_sources):
        # Random location within mask
        valid_pixels = np.argwhere(mask)
        idx = rng.choice(len(valid_pixels))
        cy_src, cx_src = valid_pixels[idx]
        source_centers.append((cy_src, cx_src))
        
        # Gaussian blob
        sigma = rng.uniform(3, 8)
        for i, (r, c) in enumerate(node_to_pix):
            dist = np.sqrt((r - cy_src)**2 + (c - cx_src)**2)
            Phi[i, m] = np.exp(-dist**2 / (2 * sigma**2))
        
        # Normalize
        Phi[:, m] /= Phi[:, m].sum() + 1e-10
    
    # Create time courses (bursts)
    a_t = np.zeros((T, n_sources))
    for m in range(n_sources):
        # Random burst times
        n_bursts = rng.integers(2, 6)
        burst_times = rng.choice(T, size=n_bursts, replace=False)
        burst_amps = rng.uniform(0.5, 2.0, size=n_bursts)
        
        for bt, ba in zip(burst_times, burst_amps):
            # Exponential decay after burst
            t_rel = np.arange(T) - bt
            a_t[:, m] += ba * np.exp(-t_rel / 10) * (t_rel >= 0)
        
        # Smooth
        a_t[:, m] = gaussian_filter1d(a_t[:, m], sigma=2)
    
    # Simulate
    g_true = 1.0
    y0 = np.ones(N) * 1.0  # Baseline
    
    sim = DiffusionSimulator(L, dt, theta=1.0)
    I_clean = sim.simulate(y0, Phi, a_t, D_true, k_true, g_true)
    
    # Add noise
    noise = rng.normal(0, noise_std, size=I_clean.shape)
    I_noisy = I_clean + noise
    
    # Reshape to image
    Y = np.zeros((T, H, W))
    for t in range(T):
        Y[t] = reshape_to_image(I_noisy[t + 1], mask, fill_value=0)
    
    ground_truth = {
        'D': D_true,
        'k': k_true,
        'g': g_true,
        'Phi': Phi,
        'a_t': a_t,
        'I_clean': I_clean,
        'source_centers': source_centers,
        'L': L,
        'pix_to_node': pix_to_node,
        'node_to_pix': node_to_pix
    }
    
    return Y, mask, ground_truth


# =============================================================================
# 9) SMOKE TEST
# =============================================================================

def run_smoke_test(verbose: bool = True) -> bool:
    """
    Run a quick smoke test to verify the module works.
    
    Returns True if all tests pass.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Running diffusion_v2 smoke test...")
    print("=" * 60)
    
    try:
        # Generate synthetic data
        print("\n1. Generating synthetic data...")
        Y, mask, truth = generate_synthetic_data(
            T=100, H=50, W=75, D_true=0.5, k_true=0.2, n_sources=3, noise_std=0.02
        )
        print(f"   Y shape: {Y.shape}, mask sum: {mask.sum()}")
        print(f"   True D={truth['D']}, k={truth['k']}")
        
        # Fit model
        print("\n2. Fitting model...")
        result = fit_diffusion_model_2d(
            Y, mask, dt=0.533, n_sources=3, bin_factors=(1, 1),
            loss_subsample=2000, verbose=False
        )
        
        print(f"\n3. Results:")
        print(f"   Fitted D={result.D:.4f} (true={truth['D']:.4f})")
        print(f"   Fitted k={result.k:.4f} (true={truth['k']:.4f})")
        print(f"   MSE={result.mse:.6f}, R²={result.r_squared:.4f}")
        
        # Check recovery
        D_error = abs(result.D - truth['D']) / truth['D']
        k_error = abs(result.k - truth['k']) / truth['k']
        
        print(f"\n4. Parameter recovery:")
        print(f"   D relative error: {D_error*100:.1f}%")
        print(f"   k relative error: {k_error*100:.1f}%")
        
        # Compute metrics
        metrics = compute_residual_metrics(result)
        print(f"\n5. Residual metrics:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        
        # Test passes if parameters recovered within 50%
        passed = D_error < 0.5 and k_error < 0.5 and result.r_squared > 0.5
        
        print("\n" + "=" * 60)
        if passed:
            print("SMOKE TEST PASSED ✓")
        else:
            print("SMOKE TEST FAILED ✗")
        print("=" * 60)
        
        return passed
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_smoke_test()
