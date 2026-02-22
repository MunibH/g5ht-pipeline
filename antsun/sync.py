"""
sync.py — Synchronization functions for NIR and confocal timing.

Converted from Julia (BehaviorDataNIR package, sync.jl).

Usage
-----
    from sync import sync_timing_from_h5, get_timing_info

    # Quick sync from an HDF5 file:
    confocal_to_nir, nir_to_confocal, timing_stack, timing_nir = sync_timing_from_h5(path_h5, n_rec=1)

    # Full timing info (populates a data dictionary):
    param = {"n_rec": 1, "max_t": 500, "t_range": list(range(1, 501)), "FLIR_FPS": 100}
    data_dict = get_timing_info(param, path_h5, h5_confocal_time_lag=0)

Notes
-----
- HDF5 files written by Julia's HDF5.jl store matrices with reversed dimensions
  relative to Python/h5py.  A Julia (N_samples, 3) array appears as (3, N_samples)
  in h5py.  The H5-reading functions in this module auto-detect and normalize the
  layout to (N_samples, N_channels).
- ``nir_to_confocal`` stores **1-based** confocal volume indices (0 = unassigned)
  to stay compatible with downstream Julia-originated workflows.
"""

import numpy as np
import h5py
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_column_layout(arr: np.ndarray, expected_n_channels: int) -> np.ndarray:
    """Ensure a 2-D array has shape (n_samples, n_channels).

    HDF5 files written by Julia reverse the dimension order, so a Julia
    (N, C) matrix is read by h5py as (C, N).  This function transposes
    when necessary.
    """
    if (arr.ndim == 2
            and arr.shape[0] == expected_n_channels
            and arr.shape[0] < arr.shape[1]):
        return arr.T
    return arr


# ---------------------------------------------------------------------------
# filter_ai_laser
# ---------------------------------------------------------------------------

def filter_ai_laser(ai_laser: np.ndarray,
                    di_camera: np.ndarray,
                    n_rec: int = 1) -> np.ndarray:
    """Filter laser analog signal to keep only z-stack recording periods.

    Parameters
    ----------
    ai_laser : 1-D array
        Raw analog laser signal.
    di_camera : 1-D array
        Digital trigger from confocal camera.
    n_rec : int
        Number of recording epochs expected.

    Returns
    -------
    1-D float32 array — laser signal zeroed outside recording periods.
    """
    n = min(len(ai_laser), len(di_camera))
    ai_laser_zstack_only = np.array(ai_laser[:n], dtype=np.float32)
    ai_laser_filter_bit = np.zeros(n, dtype=np.float32)
    trg_state = np.zeros(n, dtype=np.float64)

    n_kernel = 100
    # Maximum filter with window ±n_kernel.
    # (Could be replaced by scipy.ndimage.maximum_filter1d for speed.)
    for i in range(n):
        start = max(0, i - n_kernel)
        stop = min(n, i + n_kernel + 1)       # +1 → exclusive endpoint
        trg_state[i] = np.max(di_camera[start:stop])

    diff_trg_state = np.diff(trg_state)
    list_idx_start = np.where(diff_trg_state == 1)[0]
    list_idx_end   = np.where(diff_trg_state == -1)[0]

    if n_rec > len(list_idx_start):
        raise ValueError(
            "filter_ai_laser: not enough recordings detected. Check n_rec."
        )

    # Keep the n_rec longest recording periods
    list_idx_rec = np.argsort(list_idx_end - list_idx_start)[::-1][:n_rec]
    for idx in list_idx_rec:
        ai_laser_filter_bit[list_idx_start[idx] + 1 : list_idx_end[idx]] = 1

    return ai_laser_filter_bit * ai_laser_zstack_only


# ---------------------------------------------------------------------------
# detect_confocal_timing
# ---------------------------------------------------------------------------

def detect_confocal_timing(ai_laser: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Detect confocal z-stack start/stop times from the filtered laser signal.

    Parameters
    ----------
    ai_laser : 1-D array
        **Filtered** analog laser signal (output of ``filter_ai_laser``).

    Returns
    -------
    (list_stack_start, list_stack_stop) — sample indices for each z-stack.
    """
    ai_laser_bin = (ai_laser > np.mean(ai_laser)).astype(np.int16)

    list_confocal_on  = np.where(np.diff(ai_laser_bin) ==  1)[0] + 1
    list_confocal_off = np.where(np.diff(ai_laser_bin) == -1)[0] + 1

    # Stack boundaries: gaps > 150 samples between consecutive on/off events
    list_stack_start = list_confocal_on[
        np.where(np.diff(list_confocal_on) > 150)[0] + 1
    ]
    list_stack_start = np.insert(list_stack_start, 0, list_confocal_on[0])

    list_stack_stop = list_confocal_off[
        np.where(np.diff(list_confocal_off) > 150)[0]
    ]
    list_stack_stop = np.append(list_stack_stop, list_confocal_off[-1])

    if len(list_stack_start) != len(list_stack_stop):
        raise ValueError("n(stack_off_confocal) != n(stack_on_confocal)")

    list_stack_diff = list_stack_stop - list_stack_start
    idx_vol = np.arange(len(list_stack_diff))

    # Discard last volume if it is significantly shorter (truncated)
    if len(list_stack_diff) > 1 and np.diff(list_stack_diff)[-1] < -3:
        idx_vol = np.arange(len(list_stack_diff) - 1)

    return list_stack_start[idx_vol], list_stack_stop[idx_vol]

def detect_confocal_timing_di(di_confocal: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Detect confocal z-stack start/stop times from the digital input of the confocal camera.

    Parameters
    ----------
    di_confocal : 1-D array
        Digital input from confocal camera.

    Returns
    -------
    (list_stack_start, list_stack_stop) — sample indices for each z-stack.
    """
    di_confocal_bin = (di_confocal > np.mean(di_confocal)).astype(np.int16)

    list_confocal_on  = np.where(np.diff(di_confocal_bin) ==  1)[0] + 1
    list_confocal_off = np.where(np.diff(di_confocal_bin) == -1)[0] + 1

    # # Stack boundaries: gaps > 150 samples between consecutive on/off events
    # list_stack_start = list_confocal_on[
    #     np.where(np.diff(list_confocal_on) > 150)[0] + 1
    # ]
    # list_stack_start = np.insert(list_stack_start, 0, list_confocal_on[0])
    list_stack_start = list_confocal_on

    # list_stack_stop = list_confocal_off[
    #     np.where(np.diff(list_confocal_off) > 150)[0]
    # ]
    # list_stack_stop = np.append(list_stack_stop, list_confocal_off[-1])
    list_stack_stop = list_confocal_off

    if len(list_stack_start) != len(list_stack_stop):
        raise ValueError("n(stack_off_confocal) != n(stack_on_confocal)")

    list_stack_diff = list_stack_stop - list_stack_start
    idx_vol = np.arange(len(list_stack_diff))

    # Discard last volume if it is significantly shorter (truncated)
    if len(list_stack_diff) > 1 and np.diff(list_stack_diff)[-1] < -3:
        idx_vol = np.arange(len(list_stack_diff) - 1)

    return list_stack_start[idx_vol], list_stack_stop[idx_vol]


# ---------------------------------------------------------------------------
# detect_nir_timing
# ---------------------------------------------------------------------------

def detect_nir_timing(di_nir: np.ndarray,
                      img_id: np.ndarray,
                      q_iter_save: np.ndarray,
                      n_img_nir: int) -> np.ndarray:
    """Detect start/stop sample indices for each saved NIR camera frame.

    Parameters
    ----------
    di_nir : 1-D array
        Digital trigger from NIR (FLIR) camera.
    img_id : 1-D array
        Image IDs from metadata.
    q_iter_save : 1-D bool/int array
        Whether each iteration was saved.
    n_img_nir : int
        Number of NIR images actually saved.

    Returns
    -------
    (n_img_nir, 2) array with columns ``[nir_on, nir_off]`` (sample indices).
    """
    list_nir_on  = np.where(np.diff(di_nir) >  1)[0] + 1
    list_nir_off = np.where(np.diff(di_nir) < -1)[0] + 1
    nir_record_on  = np.diff(list_nir_on)  > 500
    nir_record_off = np.diff(list_nir_off) > 500

    # --- recording start ---
    if list_nir_on[0] > 500:
        s_nir_start = list_nir_on[0]
    elif np.sum(nir_record_on) == 2:
        s_nir_start = list_nir_on[np.where(nir_record_on)[0][0] + 1]
    else:
        raise ValueError(
            "More than 2 recording-on transitions detected for FLIR camera"
        )

    # --- recording stop ---
    if list_nir_off[-1] < len(di_nir) - 500:
        s_nir_stop = list_nir_off[-1]
    elif np.sum(nir_record_off) <= 2:
        s_nir_stop = list_nir_off[
            np.where(np.diff(list_nir_off) > 500)[0][-1]
        ]
    else:
        raise ValueError(
            "More than 2 recording-off transitions detected for FLIR camera"
        )

    # Keep triggers inside the recording window (±5 samples tolerance)
    mask_on  = (s_nir_start - 5 < list_nir_on)  & (list_nir_on  < s_nir_stop + 5)
    mask_off = (s_nir_start - 5 < list_nir_off) & (list_nir_off < s_nir_stop + 5)
    list_nir_on  = list_nir_on[mask_on]
    list_nir_off = list_nir_off[mask_off]

    if len(list_nir_on) != len(list_nir_off):
        raise ValueError(
            f"len(list_nir_on)={len(list_nir_on)} != "
            f"len(list_nir_off)={len(list_nir_off)}"
        )

    # ----- match detected triggers with image IDs -----
    # img_id is expected to increment by 1 for each saved frame, but may have gaps for unsaved frames.
    # so img_id_diff should be mostly 1s, with occasional larger jumps corresponding to unsaved frames.  We use q_iter_save to determine which triggers correspond to saved frames, and check that the count matches n_img_nir. 
    
    img_id_diff = np.diff(img_id).tolist()
    img_id_diff.insert(0, 1)
    total_expected = int(np.sum(np.diff(img_id)))

    if abs(len(list_nir_on) - total_expected) > 3:
        raise ValueError(
            f"Detected trigger count ({len(list_nir_on)}) differs from "
            f"image-ID count ({total_expected}) by more than 3"
        )
    else:
        img_id_diff[-1] += len(list_nir_on) - total_expected - 1

    # Boolean mask: which triggers correspond to a saved frame
    idx_nir_save: list = []
    for delta_n, q_save in zip(img_id_diff, q_iter_save):
        delta_n = int(delta_n)
        idx_nir_save.append(bool(q_save))
        if delta_n > 1:
            idx_nir_save.extend([False] * (delta_n - 1))

    if sum(idx_nir_save) != n_img_nir:
        raise ValueError(
            f"Detected saved NIR frames ({sum(idx_nir_save)}) "
            f"!= expected ({n_img_nir})"
        )

    return np.column_stack((list_nir_on, list_nir_off))[np.array(idx_nir_save), :]


def detect_nir_timing_from_h5(path_h5: str) -> np.ndarray:
    """Detect NIR timing directly from an HDF5 file.

    Parameters
    ----------
    path_h5 : str
        Path to the HDF5 file.

    Returns
    -------
    (n_saved, 2) array — ``[nir_on, nir_off]`` sample indices.
    """
    with h5py.File(path_h5, 'r') as h5f:
        n_img_nir    = h5f['img_nir'].shape[0]
        daqmx_di     = h5f['daqmx_di'][:]
        img_id       = h5f['img_metadata']['img_id'][:]
        q_iter_save  = h5f['img_metadata']['q_iter_save'][:]

    daqmx_di = _ensure_column_layout(daqmx_di, 2)
    di_nir = daqmx_di[:, 1].astype(np.float32)

    return detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)

def detect_nir_timing_v2(
    di_nir: np.ndarray,
    img_id: np.ndarray,
    q_iter_save: np.ndarray,
    n_img_nir: int,
    recording_start: int,
    recording_stop: int,
    tolerance: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect NIR frame timing using a known recording window.

    Instead of inferring the recording window from large gaps in the NIR
    trigger signal, this version uses explicit ``recording_start`` /
    ``recording_stop`` sample indices (e.g. from ai_laser or di_confocal)
    to crop the NIR triggers.

    Parameters
    ----------
    di_nir : 1-D array
        Digital trigger from NIR (FLIR) camera.
    img_id : 1-D array
        Image IDs from metadata (monotonically increasing).
    q_iter_save : 1-D bool/int array
        Whether each acquisition-loop iteration was saved.
    n_img_nir : int
        Number of NIR images actually saved to disk.
    recording_start : int
        Sample index where the confocal recording begins.
    recording_stop : int
        Sample index where the confocal recording ends.
    tolerance : int
        Slack (samples) around the recording window edges.

    Returns
    -------
    timing_nir_saved : (n_img_nir, 2) array
        ``[nir_on, nir_off]`` sample indices for saved frames only.
    timing_nir_all : (n_triggers_in_window, 2) array
        ``[nir_on, nir_off]`` for every trigger in the recording window.
    """
    # ---- 1. Find ALL rising / falling edges in the NIR trigger ----
    all_nir_on  = np.where(np.diff(di_nir) >  1)[0] + 1
    all_nir_off = np.where(np.diff(di_nir) < -1)[0] + 1

    # ---- 2. Crop to the known recording window ----
    mask_on  = (all_nir_on  >= recording_start - tolerance) & (all_nir_on  <= recording_stop + tolerance)
    mask_off = (all_nir_off >= recording_start - tolerance) & (all_nir_off <= recording_stop + tolerance)
    list_nir_on  = all_nir_on[mask_on]
    list_nir_off = all_nir_off[mask_off]

    # Sanity: equal number of on/off edges
    if len(list_nir_on) != len(list_nir_off):
        raise ValueError(
            f"Unequal on/off edges after cropping: "
            f"{len(list_nir_on)} on vs {len(list_nir_off)} off"
        )

    n_triggers = len(list_nir_on)
    print(f"NIR triggers in recording window: {n_triggers}")

    # ---- 3. Build the save mask from img_id + q_iter_save ----
    #   img_id_diff[k] tells how many camera triggers elapsed between
    #   iteration k-1 and iteration k.  Usually 1; >1 means triggers
    #   fired but were not part of a saved iteration.
    img_id_diff = np.diff(img_id).tolist()
    img_id_diff.insert(0, 1)                       # first iteration = 1 trigger
    total_expected = int(np.sum(np.diff(img_id)))   # total triggers implied by metadata

    mismatch = n_triggers - total_expected
    print(f"Triggers detected: {n_triggers},  expected from img_id: {total_expected},  mismatch: {mismatch}")

    # Absorb any small mismatch into the last img_id_diff entry.
    # This accounts for a few extra/missing triggers at the very end.
    img_id_diff[-1] += mismatch - 1

    # Ensure the corrected last entry is at least 1
    if img_id_diff[-1] < 1:
        print(f"  ⚠ Warning: corrected img_id_diff[-1] = {img_id_diff[-1]}, clamping to 1")
        img_id_diff[-1] = 1

    # ---- 4. Expand into a per-trigger boolean mask ----
    #   For each acquisition iteration, the first trigger gets q_iter_save;
    #   any additional triggers (delta > 1) are marked False (unsaved).
    idx_nir_save: list = []
    for delta_n, q_save in zip(img_id_diff, q_iter_save):
        delta_n = int(delta_n)
        idx_nir_save.append(bool(q_save))
        if delta_n > 1:
            idx_nir_save.extend([False] * (delta_n - 1))

    # Trim or pad so the mask length matches the trigger count
    if len(idx_nir_save) > n_triggers:
        print(f"  ⚠ Trimming save mask from {len(idx_nir_save)} to {n_triggers}")
        idx_nir_save = idx_nir_save[:n_triggers]
    elif len(idx_nir_save) < n_triggers:
        print(f"  ⚠ Padding save mask from {len(idx_nir_save)} to {n_triggers} (extra triggers marked unsaved)")
        idx_nir_save.extend([False] * (n_triggers - len(idx_nir_save)))

    idx_nir_save = np.array(idx_nir_save, dtype=bool)
    n_saved = int(np.sum(idx_nir_save))
    print(f"Saved frames from mask: {n_saved},  expected (n_img_nir): {n_img_nir}")

    if n_saved != n_img_nir:
        raise ValueError(
            f"Saved frame count mismatch: mask says {n_saved}, "
            f"file has {n_img_nir}"
        )

    # ---- 5. Build output arrays ----
    timing_nir_all   = np.column_stack((list_nir_on, list_nir_off))
    timing_nir_saved = timing_nir_all[idx_nir_save, :]

    return timing_nir_saved, timing_nir_all

# ---------------------------------------------------------------------------
# sync_timing
# ---------------------------------------------------------------------------

def sync_timing(di_confocal: np.ndarray,
                ai_laser: np.ndarray,
                di_nir: np.ndarray,
                img_id: np.ndarray,
                q_iter_save: np.ndarray,
                n_img_nir: int) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Synchronize confocal and NIR timing.

    Parameters
    ----------
    di_confocal : 1-D array
        Digital trigger from confocal microscope.
    ai_laser : 1-D array
        Filtered analog laser signal (from ``filter_ai_laser``).
    di_nir : 1-D array
        Digital trigger from NIR camera.
    img_id, q_iter_save, n_img_nir
        Metadata — see ``detect_nir_timing``.

    Returns
    -------
    confocal_to_nir : list[np.ndarray]
        For each confocal volume, the indices of NIR frames within it.
    nir_to_confocal : 1-D int array
        For each NIR frame, the **1-based** confocal volume index (0 = unassigned).
    timing_stack : (n_volumes, 2) array
        Confocal stack ``[start, stop]`` sample indices.
    timing_nir : (n_nir, 2) array
        NIR frame ``[on, off]`` sample indices.
    """
    # timing_stack = np.column_stack(detect_confocal_timing(ai_laser))
    timing_stack = np.column_stack(detect_confocal_timing_di(di_confocal))
    timing_nir   = detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)

    confocal_to_nir  = []
    nir_to_confocal  = np.zeros(timing_nir.shape[0], dtype=int)

    for i in range(timing_stack.shape[0]):
        start_, end_ = timing_stack[i, :]

        nir_on_in  = (start_ < timing_nir[:, 0]) & (timing_nir[:, 0] < end_)
        nir_off_in = (start_ < timing_nir[:, 1]) & (timing_nir[:, 1] < end_)

        idx_ = np.where(nir_on_in & nir_off_in)[0]
        confocal_to_nir.append(idx_)
        nir_to_confocal[idx_] = i + 1            # 1-based

    return confocal_to_nir, nir_to_confocal, timing_stack, timing_nir


def sync_timing_from_h5(path_h5: str,
                        n_rec: int = 1) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Synchronize NIR and confocal timing from an HDF5 file.

    Parameters
    ----------
    path_h5 : str
        Path to the HDF5 file.
    n_rec : int
        Number of recording epochs in the file.

    Returns
    -------
    Same as ``sync_timing``.
    """
    with h5py.File(path_h5, 'r') as h5f:
        n_img_nir    = h5f['img_nir'].shape[0]
        daqmx_ai     = h5f['daqmx_ai'][:]
        daqmx_di     = h5f['daqmx_di'][:]
        img_id       = h5f['img_metadata']['img_id'][:]
        q_iter_save  = h5f['img_metadata']['q_iter_save'][:]

    # Normalize to (n_samples, n_channels)
    daqmx_ai = _ensure_column_layout(daqmx_ai, 3)
    daqmx_di = _ensure_column_layout(daqmx_di, 2)

    ai_laser = filter_ai_laser(daqmx_ai[:, 0], daqmx_di[:, 0], n_rec)
    di_nir   = daqmx_di[:, 1].astype(np.float32)
    di_confocal = daqmx_di[:, 0].astype(np.float32)

    return sync_timing(di_confocal, ai_laser, di_nir, img_id, q_iter_save, n_img_nir)


# ---------------------------------------------------------------------------
# sync_stim
# ---------------------------------------------------------------------------

def sync_stim(stim: np.ndarray,
              timing_stack: np.ndarray,
              timing_nir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align a stimulus signal to confocal and NIR timing.

    Parameters
    ----------
    stim : 1-D array
        Stimulus signal at DAQ sample rate.
    timing_stack : (n_volumes, 2) array
    timing_nir : (n_nir, 2) array

    Returns
    -------
    stim_to_confocal : 1-D array — mean stimulus per confocal volume.
    stim_to_nir      : 1-D array — stimulus value at each NIR frame midpoint.
    """
    stim_to_confocal = np.array([
        np.mean(stim[int(timing_stack[i, 0]):int(timing_stack[i, 1]) + 1])
        for i in range(timing_stack.shape[0])
    ])
    nir_midpoints = np.round(np.mean(timing_nir, axis=1)).astype(int)
    stim_to_nir = stim[nir_midpoints]

    return stim_to_confocal, stim_to_nir


# ---------------------------------------------------------------------------
# signal_stack_repeatability
# ---------------------------------------------------------------------------

def signal_stack_repeatability(signal: np.ndarray,
                               timing_stack: np.ndarray,
                               sampling_rate: int = 5000
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Calculate signal repeatability across confocal stacks.

    Useful for checking z-stack or piezo repeatability.

    Parameters
    ----------
    signal : 1-D array at DAQ sample rate.
    timing_stack : (n_volumes, 2) array.
    sampling_rate : int — Hz.

    Returns
    -------
    signal_eta_u : mean signal across stacks.
    signal_eta_s : std of signal across stacks.
    list_t       : time axis (seconds).
    n_stack      : number of stacks.
    """
    s_start = timing_stack[:, 0].astype(int)
    s_end   = timing_stack[:, 1].astype(int)
    n_stack = len(s_end)
    n_stack_len = int(np.min(s_end - s_start))

    signal_eta = np.zeros((n_stack, n_stack_len))
    for i in range(n_stack):
        signal_eta[i, :] = signal[s_start[i]:s_start[i] + n_stack_len]

    signal_eta_u = np.mean(signal_eta, axis=0)
    signal_eta_s = np.std(signal_eta, axis=0)
    list_t = np.arange(1, n_stack_len + 1) / sampling_rate

    return signal_eta_u, signal_eta_s, list_t, n_stack


# ---------------------------------------------------------------------------
# nir_vec_to_confocal
# ---------------------------------------------------------------------------

def nir_vec_to_confocal(vec: np.ndarray,
                        confocal_to_nir: list,
                        confocal_len: int) -> np.ndarray:
    """Bin NIR data to confocal time resolution.

    Parameters
    ----------
    vec : 1-D or 2-D array (if 2-D, time is on columns).
    confocal_to_nir : list[np.ndarray] — mapping from confocal volumes to NIR indices.
    confocal_len : int — number of confocal time points.

    Returns
    -------
    Array with the time axis reduced to ``confocal_len``.
    """
    if vec.ndim == 1:
        new_data = np.array([
            np.mean(vec[confocal_to_nir[t]])
            if len(confocal_to_nir[t]) > 0 else 0.0
            for t in range(confocal_len)
        ])
    elif vec.ndim == 2:
        new_data = np.zeros((vec.shape[0], confocal_len))
        for dim in range(vec.shape[0]):
            new_data[dim, :] = nir_vec_to_confocal(
                vec[dim, :], confocal_to_nir, confocal_len
            )
    else:
        raise ValueError("Vector dimension cannot be greater than 2.")
    return new_data


# ---------------------------------------------------------------------------
# unlag_vec
# ---------------------------------------------------------------------------

def unlag_vec(vec: np.ndarray, lag: int) -> np.ndarray:
    """Shift a lagged vector to correct for the lag, padding with NaN.

    Parameters
    ----------
    vec : 1-D or 2-D array.
    lag : int — lag in samples.
    """
    if vec.ndim == 1:
        unlagged = np.full(len(vec) + lag, np.nan)
        unlagged[lag // 2 : lag // 2 + len(vec)] = vec
    elif vec.ndim == 2:
        unlagged = np.full((vec.shape[0], vec.shape[1] + lag), np.nan)
        for dim in range(vec.shape[0]):
            unlagged[dim, :] = unlag_vec(vec[dim, :], lag)
    else:
        raise ValueError("Vector dimension cannot be greater than 2.")
    return unlagged


# ---------------------------------------------------------------------------
# nir_to_confocal_t
# ---------------------------------------------------------------------------

def nir_to_confocal_t(t: int, nir_to_confocal: np.ndarray) -> int:
    """Convert an NIR frame index to the nearest confocal volume index.

    Searches backward from *t* until a mapped confocal volume is found.

    Parameters
    ----------
    t : int — 0-based NIR frame index.
    nir_to_confocal : 1-D int array (1-based values; 0 = unassigned).

    Returns
    -------
    int — 1-based confocal volume index.
    """
    for t_check in range(t, -1, -1):
        if nir_to_confocal[t_check] > 0:
            return int(nir_to_confocal[t_check])
    return 1


# ---------------------------------------------------------------------------
# get_timestamps
# ---------------------------------------------------------------------------

def get_timestamps(path_h5: str) -> np.ndarray:
    """Return NIR timestamps (seconds) for saved frames only.

    Parameters
    ----------
    path_h5 : str — path to the HDF5 file.
    """
    with h5py.File(path_h5, 'r') as f:
        timestamps = f['img_metadata']['img_timestamp'][:]
        saving     = f['img_metadata']['q_iter_save'][:].astype(bool)
    return timestamps[saving] / 1e9


# ---------------------------------------------------------------------------
# pre_confocal_timesteps
# ---------------------------------------------------------------------------

def pre_confocal_timesteps(data_dict: dict,
                           param: dict) -> Tuple[np.ndarray, list]:
    """Compute confocal timesteps backward from the start of confocal recording.

    Parameters
    ----------
    data_dict : must contain ``avg_timestep`` and ``nir_to_confocal``.
    param     : must contain ``FLIR_FPS``.

    Returns
    -------
    pre_nir_to_conf  : 1-D array — pre-recording NIR→confocal map.
    pre_conf_to_nir  : list[list[int]] — pre-recording confocal→NIR map.
    """
    step = data_dict["avg_timestep"] * param["FLIR_FPS"]
    # Number of NIR frames before the first confocal volume
    idx = int(np.where(data_dict["nir_to_confocal"] == 1)[0][0])

    pre_nir_to_conf = np.zeros(idx)
    n_prev = int(np.floor(idx / step))
    pre_conf_to_nir: list = []
    offset = idx - int(np.floor(step * n_prev))

    for t in range(1, n_prev + 1):
        rng_start = int(np.floor((t - 1) * step + offset))
        rng_end   = int(np.floor(t * step + offset))      # exclusive
        pre_nir_to_conf[rng_start:rng_end] = t
        pre_conf_to_nir.append(list(range(rng_start, rng_end)))

    return pre_nir_to_conf, pre_conf_to_nir


# ---------------------------------------------------------------------------
# get_timing_info  (high-level orchestrator)
# ---------------------------------------------------------------------------

def get_timing_info(param: dict,
                    path_h5: str,
                    h5_confocal_time_lag: int = 0) -> dict:
    """Initialise all timing / sync variables into a dictionary.

    Parameters
    ----------
    param : dict
        Must contain the keys:
        - ``n_rec``    : int — number of recordings.
        - ``max_t``    : int — number of confocal time points to use.
        - ``t_range``  : list/array — e.g. ``list(range(1, max_t+1))``.
        - ``FLIR_FPS`` : float — FLIR camera frame rate.
    path_h5 : str
        Path to the HDF5 file.
    h5_confocal_time_lag : int
        Confocal time-lag offset to trim.

    Returns
    -------
    dict with all timing variables populated.
    """
    data_dict: dict = {}

    # --- core sync ---
    c2n, n2c, t_stack, t_nir = sync_timing_from_h5(path_h5, param["n_rec"])

    if h5_confocal_time_lag == 0:
        c2n     = c2n[:param["max_t"]]
        t_stack = t_stack[:param["max_t"]]
        n2c     = np.array([0.0 if x > param["max_t"] else x for x in n2c])
    else:
        lag = h5_confocal_time_lag
        c2n     = c2n[lag:]
        t_stack = t_stack[lag:]
        n2c     = np.array([
            0.0 if x < lag + 1 else x - lag for x in n2c
        ])

    data_dict["confocal_to_nir"]  = c2n
    data_dict["nir_to_confocal"]  = n2c
    data_dict["timing_stack"]     = t_stack
    data_dict["timing_nir"]       = t_nir

    # --- timestamps ---
    data_dict["nir_timestamps"] = get_timestamps(path_h5)
    _v2c = lambda v: nir_vec_to_confocal(v, data_dict["confocal_to_nir"],
                                         param["max_t"])
    data_dict["timestamps"] = _v2c(data_dict["nir_timestamps"])
    data_dict["max_t_nir"]  = len(data_dict["nir_to_confocal"])
    data_dict["avg_timestep"] = (
        (data_dict["timestamps"][-1] - data_dict["timestamps"][0])
        / len(data_dict["timestamps"])
    )

    # --- pre-confocal timesteps ---
    data_dict["pre_nir_to_confocal"], data_dict["pre_confocal_to_nir"] = \
        pre_confocal_timesteps(data_dict, param)
    data_dict["max_t"]       = param["max_t"]
    data_dict["t_range"]     = param["t_range"]
    data_dict["pre_max_t"]   = len(data_dict["pre_confocal_to_nir"])
    data_dict["pre_t_range"] = list(range(1, data_dict["pre_max_t"] + 1))

    # --- stimulus detection ---
    with h5py.File(path_h5, 'r') as f:
        daqmx_ai = f['daqmx_ai'][:]
    daqmx_ai = _ensure_column_layout(daqmx_ai, 3)
    stim = daqmx_ai[:, 2]

    timing_nir_h5 = detect_nir_timing_from_h5(path_h5)
    thresh = max(float(np.max(stim)) / 2, 0.1)
    nir_midpoints = np.round(np.mean(timing_nir_h5, axis=1)).astype(int)

    stim_to_nir = np.where(stim[nir_midpoints] > thresh)[0]
    stim_to_nir_set = set(stim_to_nir.tolist())
    prev_len = 20
    stim_begin_nir = [
        t for t in stim_to_nir
        if not any(t - i in stim_to_nir_set for i in range(1, prev_len + 1))
    ]
    stim_to_confocal = [
        int(np.max(data_dict["nir_to_confocal"][:s + 1]))
        for s in stim_begin_nir
    ]
    data_dict["stim_begin_nir"]      = stim_begin_nir
    data_dict["stim_begin_confocal"] = stim_to_confocal

    return data_dict


# ---------------------------------------------------------------------------
# fill_timeskip
# ---------------------------------------------------------------------------

def fill_timeskip(traces: np.ndarray,
                  timestamps: np.ndarray,
                  min_timeskip_length: float = 5.0,
                  timeskip_step: float = 1.0,
                  fill_val: float = 0.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Insert fill points inside large temporal gaps for easier visualisation.

    Parameters
    ----------
    traces : (n_traces, n_timepoints) array.
    timestamps : 1-D array (seconds).
    min_timeskip_length : float — gap threshold (s).
    timeskip_step : float — seconds per inserted fill point.
    fill_val : float — value used for fill points.

    Returns
    -------
    (new_timestamps, new_traces) — with fill points inserted.
    """
    timeskips = np.where(np.diff(timestamps) >= min_timeskip_length)[0]
    if len(timeskips) == 0:
        return timestamps.copy(), traces.copy()

    n_traces = traces.shape[0]
    new_traces = [[] for _ in range(n_traces)]
    new_timestamps: list = []
    prev_end = 0                       # start of the next un-copied segment

    for ts in timeskips:
        num_steps = int(np.floor(
            (timestamps[ts + 1] - timestamps[ts]) / timeskip_step
        ))
        for n in range(n_traces):
            new_traces[n].extend(traces[n, prev_end:ts + 1].tolist())
            new_traces[n].extend([fill_val] * num_steps)
        new_timestamps.extend(timestamps[prev_end:ts + 1].tolist())
        new_timestamps.extend([
            timestamps[ts] + t * timeskip_step
            for t in range(1, num_steps + 1)
        ])
        prev_end = ts + 1              # NOTE: fixed — Julia original did not update

    # Remaining data after last timeskip
    for n in range(n_traces):
        new_traces[n].extend(traces[n, prev_end:].tolist())
    new_timestamps.extend(timestamps[prev_end:].tolist())

    return np.array(new_timestamps), np.array(new_traces)


def fill_timeskip_behavior(behavior: np.ndarray,
                           timestamps: np.ndarray,
                           min_timeskip_length: float = 5.0,
                           timeskip_step: float = 1.0,
                           fill_val: float = np.nan
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around ``fill_timeskip`` for a 1-D behavioural signal.

    Returns ``(new_timestamps, new_behavior_1d)``.
    """
    vec = behavior.reshape(1, -1)
    new_ts, new_vec = fill_timeskip(
        vec, timestamps,
        min_timeskip_length=min_timeskip_length,
        timeskip_step=timeskip_step,
        fill_val=fill_val,
    )
    return new_ts, new_vec[0, :]
