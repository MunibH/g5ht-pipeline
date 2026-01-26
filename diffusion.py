import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from sklearn.decomposition import NMF


# ----------------------------
# 1) Optional spatial binning
# ----------------------------
def bin2d(arr2d, by_h=2, by_w=2, mode="mean"):
    """Bin a 2D array by integer factors (by_h, by_w)."""
    H, W = arr2d.shape
    H2, W2 = (H // by_h) * by_h, (W // by_w) * by_w
    x = arr2d[:H2, :W2]
    x = x.reshape(H2 // by_h, by_h, W2 // by_w, by_w)
    if mode == "mean":
        return x.mean(axis=(1, 3))
    elif mode == "max":
        return x.max(axis=(1, 3))
    else:
        raise ValueError("mode must be 'mean' or 'max'")

def bin_movie(Y, by_h=2, by_w=2, mode="mean"):
    """Bin a (T,H,W) movie."""
    T, H, W = Y.shape
    out = []
    for t in range(T):
        out.append(bin2d(Y[t], by_h, by_w, mode=mode))
    return np.stack(out, axis=0)


# ---------------------------------------
# 2) Mask-graph Laplacian (sparse)
# ---------------------------------------
def masked_laplacian_2d(mask):
    """
    Build a 4-neighbor graph Laplacian on the True pixels of `mask`.
    This naturally corresponds to no-flux at the mask boundary (neighbors outside mask absent).

    Returns
    -------
    L : (N,N) csr sparse Laplacian
    pix_to_node : int array (H,W) mapping pixel->node index, -1 for outside
    node_to_pix : (N,2) array of (r,c) for each node
    """
    mask = mask.astype(bool)
    H, W = mask.shape
    pix_to_node = -np.ones((H, W), dtype=np.int32)
    coords = np.argwhere(mask)
    N = coords.shape[0]
    for idx, (r, c) in enumerate(coords):
        pix_to_node[r, c] = idx

    rows, cols, data = [], [], []
    for i, (r, c) in enumerate(coords):
        deg = 0
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                j = pix_to_node[rr, cc]
                rows.append(i); cols.append(j); data.append(1.0)
                deg += 1
        rows.append(i); cols.append(i); data.append(-deg)

    L = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    node_to_pix = coords
    return L, pix_to_node, node_to_pix


# ---------------------------------------
# 3) Data-driven sources via NMF
# ---------------------------------------
def learn_sources_nmf(Y, mask, n_sources=8, hp_sigma_frames=3.0, nmf_max_iter=500, seed=0):
    """
    Learn data-driven spatial source maps (Phi) and time courses (A) using NMF
    on rectified high-pass activity.

    Parameters
    ----------
    Y : (T,H,W) float, ΔF/F0
    mask : (H,W) bool
    n_sources : M
    hp_sigma_frames : temporal smoothing sigma in frames (for high-pass)
    """
    T, H, W = Y.shape
    mask = mask.astype(bool)

    # flatten masked pixels
    Yv = Y[:, mask]  # (T, N)

    # high-pass in time (smooth then subtract)
    Y_smooth = gaussian_filter1d(Yv, sigma=hp_sigma_frames, axis=0, mode="nearest")
    A = Yv - Y_smooth

    # rectify to focus on upward "release-like" transients
    A = np.maximum(A, 0.0)

    # NMF expects nonnegative samples x features: (T, N)
    nmf = NMF(
        n_components=n_sources,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=nmf_max_iter,
        random_state=seed,
    )
    W_time = nmf.fit_transform(A)      # (T, M)
    H_spat = nmf.components_           # (M, N)

    # Phi: (N, M) spatial basis; a_t: (T, M) time courses
    Phi = H_spat.T
    a_t = W_time
    return Phi, a_t


# ---------------------------------------
# 4) Semi-implicit simulator on masked nodes
# ---------------------------------------
def simulate_rd_masked(y0, Phi, a_t, L, dt, D, k, g=1.0):
    """
    Semi-implicit step:
      (I - dt*(D*L - k*I)) i_{t+1} = i_t + dt * g*Phi a_t

    y0 : (N,) initial field on masked nodes
    Phi : (N,M)
    a_t : (T,M)
    L : (N,N) Laplacian on masked nodes
    """
    N = y0.size
    I = sp.eye(N, format="csr")
    Aop = D * L - (k * I)

    Mmat = I - dt * Aop  # (N,N)
    # factorize for fast repeated solves
    solve = spla.factorized(Mmat.tocsc())

    T = a_t.shape[0]
    out = np.zeros((T + 1, N), dtype=np.float64)
    out[0] = y0

    for t in range(T):
        src = g * (Phi @ a_t[t])       # (N,)
        rhs = out[t] + dt * src
        out[t + 1] = solve(rhs)

    return out  # (T+1, N)


# ---------------------------------------
# 5) Fit D, k, g (sources fixed)
# ---------------------------------------
def fit_D_k_g(Y, mask, Phi, a_t, dt, bin_subsample=None, D0=0.5, k0=0.2, g0=1.0):
    """
    Fit scalar D,k and source gain g by minimizing SSE between simulated field and Y on masked pixels.

    bin_subsample: None or int; if set, uses a random subset of masked nodes for the loss
                   (simulation still runs on all nodes; if you want faster simulation, bin spatially first)
    """
    mask = mask.astype(bool)
    Yv = Y[:, mask]             # (T, N)
    y0 = Yv[0].astype(np.float64)
    T, N = Yv.shape

    L, _, _ = masked_laplacian_2d(mask)

    # pick loss indices
    if bin_subsample is not None and bin_subsample < N:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=bin_subsample, replace=False)
    else:
        idx = slice(None)

    def loss(theta):
        # positivity via exp
        D = np.exp(theta[0])
        k = np.exp(theta[1])
        g = np.exp(theta[2])

        Ihat = simulate_rd_masked(y0, Phi, a_t, L, dt, D, k, g=g)  # (T+1,N)
        resid = Ihat[1:] - Yv[1:]  # (T, N)
        r = resid[:, idx]
        return float(np.sum(r * r))

    theta0 = np.log([D0, k0, g0])
    res = minimize(loss, theta0, method="L-BFGS-B")
    D_hat, k_hat, g_hat = np.exp(res.x)
    return {"D": D_hat, "k": k_hat, "g": g_hat, "opt": res, "L": L}


# ---------------------------------------
# 6) End-to-end convenience wrapper
# ---------------------------------------
def fit_diffusion_model_2d(
    Y, mask, dt=0.533,
    n_sources=8,
    bin_factors=(2, 2),              # recommended start
    hp_sigma_frames=3.0,
    nmf_max_iter=500,
    loss_subsample=15000,            # random pixels for loss; set None to use all
):
    """
    Full starter pipeline:
      - optional binning
      - learn Phi,a_t via NMF
      - fit D,k,g with sources fixed
      - return reconstruction on the (possibly binned) grid

    Returns dict with fitted params and reconstructed movie on masked nodes.
    """
    Y_in = Y
    mask_in = mask

    # Optional binning to speed everything up
    by_h, by_w = bin_factors
    if (by_h, by_w) != (1, 1):
        Yb = bin_movie(Y_in, by_h=by_h, by_w=by_w, mode="mean")
        mb = bin2d(mask_in.astype(float), by_h=by_h, by_w=by_w, mode="mean") > 0.5
    else:
        Yb, mb = Y_in, mask_in.astype(bool)

    # Learn sources
    Phi, a_t = learn_sources_nmf(
        Yb, mb, n_sources=n_sources,
        hp_sigma_frames=hp_sigma_frames,
        nmf_max_iter=nmf_max_iter
    )

    # Fit D,k,g
    fit = fit_D_k_g(Yb, mb, Phi, a_t, dt, bin_subsample=loss_subsample)

    # Reconstruct using fitted params
    Yv = Yb[:, mb]
    y0 = Yv[0].astype(np.float64)
    Ihat = simulate_rd_masked(y0, Phi, a_t, fit["L"], dt, fit["D"], fit["k"], g=fit["g"])  # (T+1,N)

    return {
        "Y_used": Yb,
        "mask_used": mb,
        "Phi": Phi,          # (N,M)
        "a_t": a_t,          # (T,M)
        "params": {"D": fit["D"], "k": fit["k"], "g": fit["g"]},
        "Ihat_masked": Ihat, # (T+1, N) on masked nodes
        "opt": fit["opt"],
        "bin_factors": bin_factors,
        "dt": dt,
    }
