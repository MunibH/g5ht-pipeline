"""Posture similarity analysis from oriented splines.

Encodes body posture as the **tangent angle profile** along the arc-length-
resampled spline.  This representation is intrinsically translation- and
rotation-invariant: only the *shape* of the body matters, not where or in
which direction the worm happens to be moving.

The tangent angle profile is then reduced via PCA — typically PC1–PC2
capture >90 % of variance — and the low-dimensional scores are used for:
  * pairwise similarity / distance matrices,
  * t-SNE embedding,
  * agglomerative clustering.

Usage from pipeline.ipynb:
    import posture_similarity
    posture_similarity.main(PTH)

Or from command line:
    uv run python posture_similarity.py /path/to/dataset_dir
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Spline loading & resampling
# ---------------------------------------------------------------------------

def load_oriented_splines(dataset_dir: str) -> dict:
    """Load oriented.json and return {frame_int: (N, 2) ndarray or None}.

    Frames with empty splines are stored as None.
    """
    oriented_path = os.path.join(dataset_dir, 'oriented.json')
    if not os.path.exists(oriented_path):
        raise FileNotFoundError(f"oriented.json not found in {dataset_dir}")

    with open(oriented_path) as f:
        raw = json.load(f)

    splines = {}
    for k, v in raw.items():
        frame = int(k)
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or len(arr) < 4:
            splines[frame] = None
        else:
            splines[frame] = arr
    return splines


def resample_spline(pts: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Resample a spline to *n_points* uniformly spaced along arc length.

    Parameters
    ----------
    pts : (N, 2) array of (y, x) coordinates.
    n_points : number of output points.

    Returns
    -------
    resampled : (n_points, 2) array.
    """
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    if total_length < 1e-8:
        return np.full((n_points, 2), np.nan)

    # normalise arc length to [0, 1]
    s = cumulative / total_length
    # remove duplicate arc-length values (can happen with coincident points)
    mask = np.concatenate([[True], np.diff(s) > 0])
    s = s[mask]
    pts_clean = pts[mask]

    if len(s) < 2:
        return np.full((n_points, 2), np.nan)

    interp_y = interp1d(s, pts_clean[:, 0], kind='linear')
    interp_x = interp1d(s, pts_clean[:, 1], kind='linear')

    s_uniform = np.linspace(0, 1, n_points)
    return np.column_stack([interp_y(s_uniform), interp_x(s_uniform)])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_tangent_angle_profile(pts: np.ndarray) -> np.ndarray:
    """Tangent angle profile along the body (orientation-invariant posture).

    Computes the tangent angle at each segment, subtracts the initial angle
    (making it rotation-invariant), and unwraps to remove discontinuities.

    Parameters
    ----------
    pts : (N, 2) resampled spline (y, x).

    Returns
    -------
    profile : (N-1,) unwrapped tangent angle profile.
    """
    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 0], diffs[:, 1])
    profile = np.unwrap(angles - angles[0])
    return profile


def compute_curvature_profile(pts: np.ndarray) -> np.ndarray:
    """Signed curvature at each interior point via discrete Frenet.

    Parameters
    ----------
    pts : (N, 2) resampled spline.

    Returns
    -------
    kappa : (N-2,) signed curvature array.
    """
    dx = np.diff(pts[:, 1])
    dy = np.diff(pts[:, 0])
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    denom = (dx[:-1] ** 2 + dy[:-1] ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    kappa = (dx[:-1] * ddy - dy[:-1] * ddx) / denom
    return kappa


def procrustes_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Procrustes distance between two point sets after optimal alignment.

    Uses scipy.spatial.procrustes which standardises, aligns, and returns
    the sum of squared differences.
    """
    _, _, disparity = scipy_procrustes(pts_a, pts_b)
    return float(disparity)


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

def compute_all_features(
    splines: dict,
    n_points: int = 100,
    n_pcs: int = 10,
) -> dict:
    """Resample all splines, compute tangent angle profiles, and fit PCA.

    The primary posture representation is the **tangent angle profile**
    projected onto its top PCA components.  Raw curvature profiles are also
    stored for reference but are *not* used for clustering.

    Parameters
    ----------
    splines : dict from load_oriented_splines().
    n_points : resampling resolution (100 is typically sufficient).
    n_pcs : number of PCA components to retain.

    Returns
    -------
    features : dict with keys:
        'frames'           : sorted list of frame indices
        'resampled'        : dict {frame: (n_points, 2) or None}
        'tangent_profile'  : dict {frame: (n_points-1,) or None}
        'curvature'        : dict {frame: (n_points-2,) or None}
        'valid_mask'       : bool array aligned with 'frames'
        'pca'              : fitted PCA object
        'pca_scores'       : dict {frame: (n_pcs,) or None}
        'feature_matrix'   : (T_valid, n_points-1) raw tangent profiles for valid frames
        'score_matrix'     : (T_valid, n_pcs) PCA scores for valid frames
        'valid_frames'     : list of valid frame indices
    """
    frames = sorted(splines.keys())
    resampled = {}
    tangent_profile = {}
    curvature = {}
    pca_scores = {}
    valid = []

    # — pass 1: resample and extract tangent angle profiles —
    raw_profiles = []
    valid_frames_list = []

    for f in frames:
        pts = splines[f]
        if pts is None or len(pts) < 4:
            resampled[f] = None
            tangent_profile[f] = None
            curvature[f] = None
            pca_scores[f] = None
            valid.append(False)
            continue

        rs = resample_spline(pts, n_points)
        if np.any(np.isnan(rs)):
            resampled[f] = None
            tangent_profile[f] = None
            curvature[f] = None
            pca_scores[f] = None
            valid.append(False)
            continue

        tp = compute_tangent_angle_profile(rs)
        resampled[f] = rs
        tangent_profile[f] = tp
        curvature[f] = compute_curvature_profile(rs)
        valid.append(True)
        raw_profiles.append(tp)
        valid_frames_list.append(f)

    # — pass 2: PCA on tangent angle profiles —
    feature_matrix = np.array(raw_profiles)  # (T_valid, n_points-1)
    effective_n_pcs = min(n_pcs, feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=effective_n_pcs)
    score_matrix = pca.fit_transform(feature_matrix)  # (T_valid, n_pcs)

    for i, f in enumerate(valid_frames_list):
        pca_scores[f] = score_matrix[i]

    return {
        'frames': frames,
        'resampled': resampled,
        'tangent_profile': tangent_profile,
        'curvature': curvature,
        'valid_mask': np.array(valid),
        'pca': pca,
        'pca_scores': pca_scores,
        'feature_matrix': feature_matrix,
        'score_matrix': score_matrix,
        'valid_frames': valid_frames_list,
    }


# ---------------------------------------------------------------------------
# Pairwise similarity / distance matrices
# ---------------------------------------------------------------------------

def compute_pairwise_tangent_similarity(
    features: dict,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Pairwise Pearson correlation of tangent angle profiles → (T, T) similarity.

    This is the *primary* posture similarity metric.
    Invalid frames get NaN rows/columns.
    """
    frames = features['frames']
    T = len(frames)
    sim = np.full((T, T), np.nan)

    for i in range(T):
        tp_i = features['tangent_profile'][frames[i]]
        if tp_i is None:
            continue
        tp_i_c = tp_i - tp_i.mean()
        norm_i = np.linalg.norm(tp_i_c)
        for j in range(i, T):
            tp_j = features['tangent_profile'][frames[j]]
            if tp_j is None:
                continue
            tp_j_c = tp_j - tp_j.mean()
            norm_j = np.linalg.norm(tp_j_c)
            corr = np.dot(tp_i_c, tp_j_c) / (norm_i * norm_j + eps)
            sim[i, j] = corr
            sim[j, i] = corr

    return sim


def compute_pairwise_procrustes(
    features: dict,
) -> np.ndarray:
    """
    Pairwise Procrustes distance → (T, T) distance matrix.

    Invalid frames get NaN rows/columns.
    """
    frames = features['frames']
    T = len(frames)
    dist = np.full((T, T), np.nan)

    for i in tqdm(range(T), desc="Procrustes pairwise"):
        pts_i = features['resampled'][frames[i]]
        if pts_i is None:
            continue
        dist[i, i] = 0.0
        for j in range(i + 1, T):
            pts_j = features['resampled'][frames[j]]
            if pts_j is None:
                continue
            d = procrustes_distance(pts_i, pts_j)
            dist[i, j] = d
            dist[j, i] = d

    return dist


def compute_pairwise_pca_distance(
    features: dict,
) -> np.ndarray:
    """
    Pairwise Euclidean distance in PCA space → (T, T) distance matrix.

    Uses the top PCA scores of tangent angle profiles, which captures
    the dominant posture modes without noise.
    """
    frames = features['frames']
    T = len(frames)
    dist = np.full((T, T), np.nan)

    for i in range(T):
        sc_i = features['pca_scores'][frames[i]]
        if sc_i is None:
            continue
        dist[i, i] = 0.0
        for j in range(i + 1, T):
            sc_j = features['pca_scores'][frames[j]]
            if sc_j is None:
                continue
            d = np.linalg.norm(sc_i - sc_j)
            dist[i, j] = d
            dist[j, i] = d

    return dist


# ---------------------------------------------------------------------------
# Dimensionality reduction + clustering
# ---------------------------------------------------------------------------

def reduce_and_cluster(
    features: dict,
    n_clusters: int = 5,
    n_pcs_for_tsne: int = 10,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> tuple:
    """
    t-SNE embedding + agglomerative clustering on PCA-projected tangent angle profiles.

    Parameters
    ----------
    features : dict from compute_all_features().
    n_clusters : number of clusters.
    n_pcs_for_tsne : number of PCA components to feed into t-SNE.
    perplexity : t-SNE perplexity (auto-adjusted if T is small).
    random_state : random seed.

    Returns
    -------
    embedding : (T_valid, 2) array.
    labels : (T_valid,) cluster labels.
    valid_frames : list of frame indices corresponding to rows.
    """
    score_matrix = features['score_matrix']
    valid_frames = features['valid_frames']

    T_valid = len(valid_frames)
    if T_valid < 2:
        raise ValueError("Need at least 2 valid frames for embedding.")

    # use the available PCA scores (may be fewer than n_pcs_for_tsne)
    n_use = min(n_pcs_for_tsne, score_matrix.shape[1])
    X = score_matrix[:, :n_use]

    # adjust perplexity if necessary
    effective_perplexity = min(perplexity, (T_valid - 1) / 3.0)
    effective_perplexity = max(effective_perplexity, 2.0)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    embedding = tsne.fit_transform(X)

    # agglomerative clustering on PCA scores
    effective_n_clusters = min(n_clusters, T_valid)
    clustering = AgglomerativeClustering(n_clusters=effective_n_clusters)
    labels = clustering.fit_predict(X)

    return embedding, labels, valid_frames


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_similarity_matrix(
    matrix: np.ndarray,
    title: str = "Similarity Matrix",
    save_path: str = None,
    cmap: str = 'RdBu_r',
    vmin: float = None,
    vmax: float = None,
):
    """Heatmap of an N×N matrix."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_posture_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    valid_frames: list,
    title: str = "Posture Embedding (t-SNE)",
    save_path: str = None,
):
    """Scatter plot of the 2D embedding colored by cluster."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=labels, cmap='tab10', s=8, alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_cluster_exemplars(
    features: dict,
    labels: np.ndarray,
    valid_frames: list,
    n_per_cluster: int = 3,
    save_path: str = None,
):
    """Show representative splines from each cluster overlaid."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4), dpi=150, squeeze=False)

    for ci, cl in enumerate(unique_labels):
        ax = axes[0, ci]
        cluster_frames = [f for f, la in zip(valid_frames, labels) if la == cl]
        # pick evenly spaced exemplars
        indices = np.round(np.linspace(0, len(cluster_frames) - 1, min(n_per_cluster, len(cluster_frames)))).astype(int)
        for idx in indices:
            f = cluster_frames[idx]
            pts = features['resampled'][f]
            if pts is not None:
                ax.plot(pts[:, 1], pts[:, 0], lw=1, alpha=0.7, label=f"f{f}")
        ax.set_title(f"Cluster {cl} (n={len(cluster_frames)})")
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend(fontsize=6, loc='best')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes


def plot_posture_summary(
    tangent_sim: np.ndarray,
    pca_dist: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
):
    """Multi-panel figure: tangent similarity, PCA distance, t-SNE."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # tangent angle similarity
    im0 = axes[0].imshow(tangent_sim, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title('Tangent Angle Similarity')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Frame')

    # PCA distance
    im1 = axes[1].imshow(pca_dist, cmap='viridis', aspect='auto')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    axes[1].set_title('PCA Distance')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Frame')

    # t-SNE embedding
    sc = axes[2].scatter(embedding[:, 0], embedding[:, 1],
                         c=labels, cmap='tab10', s=8, alpha=0.7)
    plt.colorbar(sc, ax=axes[2], label='Cluster')
    axes[2].set_title('t-SNE Embedding')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(dataset_dir: str, n_points: int = 100, n_pcs: int = 10, n_clusters: int = 5):
    """End-to-end posture similarity pipeline.

    1. Load oriented splines
    2. Resample, compute tangent angle profiles, fit PCA
    3. Compute pairwise tangent similarity and PCA distance matrices
    4. t-SNE embedding + agglomerative clustering on PCA scores
    5. Save all outputs and plots

    Outputs saved to *dataset_dir*:
        - tangent_similarity.npy     (T, T) — pairwise tangent angle profile correlation
        - pca_distance.npy           (T, T) — pairwise Euclidean distance in PCA space
        - procrustes_distance.npy    (T, T) — pairwise Procrustes distance
        - posture_pca_variance.csv   — PCA explained variance ratios
        - posture_embedding.csv      frame, tsne_1, tsne_2, cluster, pc1, pc2, ...
        - posture_similarity.png     multi-panel summary
        - posture_clusters.png       cluster exemplar splines
    """
    print(f"Loading oriented splines from {dataset_dir}...")
    splines = load_oriented_splines(dataset_dir)

    print(f"Resampling {len(splines)} splines to {n_points} points, computing tangent angle profiles + PCA ({n_pcs} PCs)...")
    features = compute_all_features(splines, n_points=n_points, n_pcs=n_pcs)
    n_valid = features['valid_mask'].sum()
    print(f"  {n_valid}/{len(splines)} valid splines")

    # PCA variance summary
    pca = features['pca']
    var_df = pd.DataFrame({
        'pc': np.arange(1, pca.n_components_ + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
    })
    var_df.to_csv(os.path.join(dataset_dir, 'posture_pca_variance.csv'), index=False)
    print(f"  PCA: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC1+PC2={np.cumsum(pca.explained_variance_ratio_[:2])[-1]:.3f}")

    print("Computing pairwise tangent angle similarity...")
    tangent_sim = compute_pairwise_tangent_similarity(features)
    np.save(os.path.join(dataset_dir, 'tangent_similarity.npy'), tangent_sim)

    print("Computing pairwise PCA distance...")
    pca_dist = compute_pairwise_pca_distance(features)
    np.save(os.path.join(dataset_dir, 'pca_distance.npy'), pca_dist)

    print("Computing pairwise Procrustes distance...")
    procrustes_dist = compute_pairwise_procrustes(features)
    np.save(os.path.join(dataset_dir, 'procrustes_distance.npy'), procrustes_dist)

    print("Running t-SNE + clustering on PCA scores...")
    embedding, labels, valid_frames = reduce_and_cluster(
        features, n_clusters=n_clusters, n_pcs_for_tsne=n_pcs,
    )

    # save embedding CSV with PCA scores
    embed_data = {
        'frame': valid_frames,
        'tsne_1': embedding[:, 0],
        'tsne_2': embedding[:, 1],
        'cluster': labels,
    }
    for pc_idx in range(features['score_matrix'].shape[1]):
        embed_data[f'pc{pc_idx + 1}'] = features['score_matrix'][:, pc_idx]
    embed_df = pd.DataFrame(embed_data)
    embed_df.to_csv(os.path.join(dataset_dir, 'posture_embedding.csv'), index=False)

    # plots
    print("Generating plots...")
    plot_posture_summary(
        tangent_sim, pca_dist, embedding, labels,
        save_path=os.path.join(dataset_dir, 'posture_similarity.png'),
    )
    plt.close('all')

    plot_cluster_exemplars(
        features, labels, valid_frames,
        save_path=os.path.join(dataset_dir, 'posture_clusters.png'),
    )
    plt.close('all')

    print(f"Done. Outputs saved to {dataset_dir}")
    return features, tangent_sim, pca_dist, procrustes_dist, embedding, labels, valid_frames


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: uv run python posture_similarity.py /path/to/dataset_dir [n_clusters]")
        sys.exit(1)
    d = sys.argv[1]
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(d, n_clusters=nc)
