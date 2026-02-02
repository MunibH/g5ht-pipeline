"""
Sync NIR and Confocal Timing Module

This module provides functions to synchronize NIR camera data with confocal microscopy data,
detect timing, align signals, and process behavioral data.

Converted from Julia to Python.
"""

import numpy as np
import h5py
from typing import Tuple, List, Dict, Union
import matplotlib.pyplot as plt


def detect_nir_timing(di_nir: np.ndarray, img_id: np.ndarray, 
                      q_iter_save: np.ndarray, n_img_nir: int) -> np.ndarray:
    """
    Detect the start and stop time of the NIR camera recording.

    Parameters
    ----------
    di_nir : np.ndarray
        Digital input from the NIR camera
    img_id : np.ndarray
        Image id array
    q_iter_save : np.ndarray
        Boolean array indicating whether the image is saved
    n_img_nir : int
        Number of NIR images

    Returns
    -------
    np.ndarray
        Array of shape (n_saved_frames, 2) containing [start, stop] times for each frame
    """
    # Behavior camera - FLIR
    di_diff = np.diff(di_nir)
    list_nir_on = np.where(di_diff > 1)[0] + 1
    list_nir_off = np.where(di_diff < -1)[0] + 1
    
    nir_record_on = np.diff(list_nir_on) > 500
    nir_record_off = np.diff(list_nir_off) > 500

    # Detect recording start
    if list_nir_on[0] > 500:  # no trigger before the first
        s_nir_start = list_nir_on[0]
    elif np.sum(nir_record_on) == 2:
        s_nir_start = list_nir_on[np.argmax(nir_record_on) + 1]
    else:
        raise ValueError("more than 2 recording on detected for FLIR camera")

    # Detect recording stop
    if list_nir_off[-1] < len(di_nir) - 500:
        s_nir_stop = list_nir_off[-1]
    elif np.sum(nir_record_off) <= 2:
        idx = np.where(np.diff(list_nir_off) > 500)[0]
        if len(idx) > 0:
            s_nir_stop = list_nir_off[idx[-1]]
        else:
            s_nir_stop = list_nir_off[-1]
    else:
        raise ValueError("more than 2 recording off detected for FLIR camera")

    # Filter triggers within recording window
    list_nir_on = list_nir_on[(list_nir_on > s_nir_start - 5) & (list_nir_on < s_nir_stop + 5)]
    list_nir_off = list_nir_off[(list_nir_off > s_nir_start - 5) & (list_nir_off < s_nir_stop + 5)]

    if len(list_nir_on) != len(list_nir_off):
        raise ValueError("length(list_nir_on) != length(list_nir_off)")

    # Process image IDs
    img_id_diff = np.diff(img_id)
    img_id_diff = np.concatenate([[1], img_id_diff])
    
    if abs(len(list_nir_on) - np.sum(np.diff(img_id))) > 3:
        raise ValueError("the detected trigger count is different from the image id data by more than 3")
    else:
        img_id_diff[-1] += len(list_nir_on) - np.sum(np.diff(img_id)) - 1

    # Build index of saved NIR frames
    idx_nir_save = []
    for delta_n, q_save in zip(img_id_diff, q_iter_save):
        if delta_n == 1:
            idx_nir_save.append(q_save)
        else:
            idx_nir_save.append(q_save)
            for i in range(int(delta_n) - 1):
                idx_nir_save.append(False)

    idx_nir_save = np.array(idx_nir_save, dtype=bool)

    if np.sum(idx_nir_save) != n_img_nir:
        raise ValueError("detected number of NIR frames != saved NIR frames")

    return np.column_stack([list_nir_on, list_nir_off])[idx_nir_save, :]


def detect_nir_timing_from_file(path_h5: str) -> np.ndarray:
    """
    Detect NIR timing from HDF5 file.

    Parameters
    ----------
    path_h5 : str
        Path to HDF5 file

    Returns
    -------
    np.ndarray
        Array of NIR timing information
    """
    with h5py.File(path_h5, 'r') as h5f:
        n_img_nir = h5f['img_nir'].shape[2]
        daqmx_di = h5f['daqmx_di'][:]
        img_metadata = h5f['img_metadata']
        img_timestamp = img_metadata['img_timestamp'][:]
        img_id = img_metadata['img_id'][:]
        q_iter_save = img_metadata['q_iter_save'][:]

    di_nir = daqmx_di[:, 1].astype(np.float32)

    return detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)


def detect_confocal_timing(ai_laser: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect confocal microscopy timing from laser analog signal.

    Parameters
    ----------
    ai_laser : np.ndarray
        Laser analog input signal

    Returns
    -------
    list_stack_start : np.ndarray
        Stack start indices
    list_stack_stop : np.ndarray
        Stack stop indices
    """
    # Binarize laser analog signal
    ai_laser_bin = (ai_laser > np.mean(ai_laser)).astype(np.int16)

    ai_diff = np.diff(ai_laser_bin)
    list_confocal_on = np.where(ai_diff == 1)[0] + 1
    list_confocal_off = np.where(ai_diff == -1)[0] + 1

    # Find stack boundaries (gaps > 150 samples)
    on_diff = np.diff(list_confocal_on)
    list_stack_start = list_confocal_on[np.where(on_diff > 150)[0] + 1]
    list_stack_start = np.concatenate([[list_confocal_on[0]], list_stack_start])

    off_diff = np.diff(list_confocal_off)
    list_stack_stop = list_confocal_off[np.where(off_diff > 150)[0]]
    list_stack_stop = np.concatenate([list_stack_stop, [list_confocal_off[-1]]])

    if len(list_stack_start) != len(list_stack_stop):
        raise ValueError("n(stack_off_confocal) != n(stack_on_confocal)")

    # Check for incomplete final stack
    list_stack_diff = list_stack_stop - list_stack_start
    if len(list_stack_diff) > 1 and np.diff(list_stack_diff)[-1] < -3:
        list_stack_start = list_stack_start[:-1]
        list_stack_stop = list_stack_stop[:-1]

    return list_stack_start, list_stack_stop


def filter_ai_laser(ai_laser: np.ndarray, di_camera: np.ndarray, n_rec: int = 1) -> np.ndarray:
    """
    Filter laser signal to include only z-stack recording periods.

    Parameters
    ----------
    ai_laser : np.ndarray
        Laser analog input
    di_camera : np.ndarray
        Camera digital input
    n_rec : int, optional
        Number of recordings to keep (default: 1)

    Returns
    -------
    np.ndarray
        Filtered laser signal
    """
    n = min(len(ai_laser), len(di_camera))
    ai_laser_zstack_only = ai_laser[:n].astype(np.float32)
    ai_laser_filter_bit = np.zeros(n, dtype=np.float32)
    trg_state = np.zeros(n, dtype=np.float64)

    # validation plots
    plt.figure(figsize=(12, 6))
    plt.plot(ai_laser_zstack_only, label='Original AI Laser')
    plt.plot(di_camera[:n], label='DI Camera', alpha=0.5)
    plt.title('AI Laser and DI Camera Signals')
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.show()

    n_kernel = 100
    for i in range(n):
        start = max(0, i - n_kernel)
        stop = min(n, i + n_kernel + 1)
        trg_state[i] = np.max(di_camera[start:stop])

    delta_trg_state = np.diff(trg_state)
    list_idx_start = np.where(delta_trg_state == 1)[0]
    list_idx_end = np.where(delta_trg_state == -1)[0]

    if n_rec > len(list_idx_start):
        raise ValueError("filter_ai_laser: not enough recordings detected. check n_rec")

    # Get n_rec longest recordings
    rec_lengths = list_idx_end - list_idx_start
    list_idx_rec = np.argsort(rec_lengths)[::-1][:n_rec]
    
    for i in list_idx_rec:
        ai_laser_filter_bit[list_idx_start[i]+1:list_idx_end[i]] = 1

    return ai_laser_filter_bit * ai_laser_zstack_only


def sync_timing(di_nir: np.ndarray, ai_laser: np.ndarray, img_id: np.ndarray,
                q_iter_save: np.ndarray, n_img_nir: int) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synchronize NIR and confocal timing.

    Parameters
    ----------
    di_nir : np.ndarray
        NIR digital input
    ai_laser : np.ndarray
        Laser analog input
    img_id : np.ndarray
        Image IDs
    q_iter_save : np.ndarray
        Save flags
    n_img_nir : int
        Number of NIR images

    Returns
    -------
    confocal_to_nir : list
        List mapping confocal frames to NIR frame indices
    nir_to_confocal : np.ndarray
        Array mapping NIR frames to confocal frame index
    timing_stack : np.ndarray
        Confocal stack timing (start, stop)
    timing_nir : np.ndarray
        NIR frame timing (start, stop)
    """
    list_stack_start, list_stack_stop = detect_confocal_timing(ai_laser)
    timing_stack = np.column_stack([list_stack_start, list_stack_stop])
    timing_nir = detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir)

    confocal_to_nir = []
    nir_to_confocal = np.zeros(timing_nir.shape[0])

    for i in range(timing_stack.shape[0]):
        start_, end_ = timing_stack[i, :]

        nir_on_bit = (start_ < timing_nir[:, 0]) & (timing_nir[:, 0] < end_)
        nir_off_bit = (start_ < timing_nir[:, 1]) & (timing_nir[:, 1] < end_)

        idx_ = np.where(nir_on_bit & nir_off_bit)[0]
        confocal_to_nir.append(idx_)

        nir_to_confocal[idx_] = i + 1  # 1-indexed to match Julia behavior

    return confocal_to_nir, nir_to_confocal, timing_stack, timing_nir


def sync_timing_from_file(path_h5: str, n_rec: int = 1) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sync NIR and confocal timing from HDF5 file.

    Parameters
    ----------
    path_h5 : str
        Path to h5 file
    n_rec : int, optional
        Number of recordings in the file (default: 1)

    Returns
    -------
    confocal_to_nir : list
        List mapping confocal frames to NIR frame indices
    nir_to_confocal : np.ndarray
        Array mapping NIR frames to confocal frame index
    timing_stack : np.ndarray
        Confocal stack timing
    timing_nir : np.ndarray
        NIR frame timing
    """
    with h5py.File(path_h5, 'r') as h5f:
        n_img_nir = h5f['img_nir'].shape[2]
        daqmx_ai = h5f['daqmx_ai'][:]
        daqmx_di = h5f['daqmx_di'][:]
        img_metadata = h5f['img_metadata']
        img_timestamp = img_metadata['img_timestamp'][:]
        img_id = img_metadata['img_id'][:]
        q_iter_save = img_metadata['q_iter_save'][:]

    ai_laser = filter_ai_laser(daqmx_ai[:, 0], daqmx_di[:, 0], n_rec)
    di_nir = daqmx_di[:, 1].astype(np.float32)

    return sync_timing(di_nir, ai_laser, img_id, q_iter_save, n_img_nir)


def sync_stim(stim: np.ndarray, timing_stack: np.ndarray, 
              timing_nir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align stimulus signal to confocal and NIR timing.

    Parameters
    ----------
    stim : np.ndarray
        Stimulus signal
    timing_stack : np.ndarray
        Confocal timing
    timing_nir : np.ndarray
        NIR timing

    Returns
    -------
    stim_to_confocal : np.ndarray
        Stimulus aligned to confocal frames
    stim_to_nir : np.ndarray
        Stimulus aligned to NIR frames
    """
    stim_to_confocal = np.array([
        np.mean(stim[timing_stack[i, 0]:timing_stack[i, 1]])
        for i in range(timing_stack.shape[0])
    ])
    
    nir_mean_times = np.round(np.mean(timing_nir, axis=1)).astype(int)
    stim_to_nir = stim[nir_mean_times]

    return stim_to_confocal, stim_to_nir


def signal_stack_repeatability(signal: np.ndarray, timing_stack: np.ndarray,
                               sampling_rate: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate signal repeatability across stacks (e.g., checking z-stack repeatability).

    Parameters
    ----------
    signal : np.ndarray
        Signal array
    timing_stack : np.ndarray
        Confocal timing
    sampling_rate : int, optional
        Sampling rate in Hz (default: 5000)

    Returns
    -------
    signal_eta_u : np.ndarray
        Mean signal across stacks
    signal_eta_s : np.ndarray
        Standard deviation across stacks
    list_t : np.ndarray
        Time array in seconds
    n_stack : int
        Number of stacks
    """
    s_stack_start = timing_stack[:, 0]
    s_stack_end = timing_stack[:, 1]
    n_stack = s_stack_end.shape[0]
    n_stack_len = np.min(s_stack_end - s_stack_start)
    
    signal_eta = np.zeros((n_stack, n_stack_len))
    for i in range(n_stack):
        signal_eta[i, :] = signal[s_stack_start[i]:s_stack_start[i] + n_stack_len]

    signal_eta_u = np.mean(signal_eta, axis=0)
    signal_eta_s = np.std(signal_eta, axis=0)
    list_t = np.arange(n_stack_len) / sampling_rate

    return signal_eta_u, signal_eta_s, list_t, n_stack


def nir_vec_to_confocal(vec: np.ndarray, confocal_to_nir: List, 
                        confocal_len: int) -> np.ndarray:
    """
    Bins NIR behavioral data to match the confocal time points.

    Parameters
    ----------
    vec : np.ndarray
        Behavioral data vector. Can be 1D or 2D; if 2D time should be on the columns
    confocal_to_nir : list
        Confocal to NIR time sync
    confocal_len : int
        Length of confocal dataset

    Returns
    -------
    np.ndarray
        Behavioral data aligned to confocal timepoints
    """
    if vec.ndim == 1:
        new_data = np.array([np.mean(vec[confocal_to_nir[t]]) for t in range(confocal_len)])
    elif vec.ndim == 2:
        new_data = np.zeros((vec.shape[0], confocal_len))
        for dim in range(vec.shape[0]):
            new_data[dim, :] = nir_vec_to_confocal(vec[dim, :], confocal_to_nir, confocal_len)
    else:
        raise ValueError("Vector dimension cannot be greater than 2.")
    
    return new_data


def unlag_vec(vec: np.ndarray, lag: int) -> np.ndarray:
    """
    Shifts a lagged vector to correct for the lag amount.

    Parameters
    ----------
    vec : np.ndarray
        Lagged vector
    lag : int
        Lag amount to correct

    Returns
    -------
    np.ndarray
        Unlagged vector
    """
    if vec.ndim == 1:
        unlagged_vec = np.full(len(vec) + lag, np.nan)
        unlagged_vec[lag//2:-lag//2] = vec
    elif vec.ndim == 2:
        unlagged_vec = np.full((vec.shape[0], vec.shape[1] + lag), np.nan)
        for dim in range(vec.shape[0]):
            unlagged_vec[dim, :] = unlag_vec(vec[dim, :], lag)
    else:
        raise ValueError("Vector dimension cannot be greater than 2.")
    
    return unlagged_vec


def nir_to_confocal_t(t: int, nir_to_confocal: np.ndarray) -> int:
    """
    Converts NIR time point to confocal time point using timesync variable.

    Parameters
    ----------
    t : int
        NIR time point
    nir_to_confocal : np.ndarray
        NIR to confocal timesync array

    Returns
    -------
    int
        Confocal time point
    """
    for t_check in range(t, -1, -1):
        if nir_to_confocal[t_check] > 0:
            return int(nir_to_confocal[t_check])
    return 1


def get_timestamps(path_h5: str) -> np.ndarray:
    """
    Gets NIR timestamps from the NIR data file.

    Parameters
    ----------
    path_h5 : str
        Path to HDF5 file

    Returns
    -------
    np.ndarray
        Timestamps in seconds
    """
    with h5py.File(path_h5, 'r') as f:
        timestamps = f['img_metadata']['img_timestamp'][:]
        saving = f['img_metadata']['q_iter_save'][:]
    
    return timestamps[saving] / 1e9


def get_timing_info(data_dict: Dict, param: Dict, path_h5: str, 
                    h5_confocal_time_lag: int) -> None:
    """
    Initializes all timing and syncing variables into data_dict.

    Parameters
    ----------
    data_dict : dict
        Dictionary to store timing information
    param : dict
        Parameters dictionary containing 'n_rec', 'max_t', 't_range', 'FLIR_FPS'
    path_h5 : str
        Path to HDF5 file
    h5_confocal_time_lag : int
        Time lag for confocal data

    Returns
    -------
    None
        Modifies data_dict in place
    """
    # Get basic timing sync
    confocal_to_nir, nir_to_confocal, timing_stack, timing_nir = sync_timing_from_file(path_h5, param['n_rec'])
    
    # Apply time lag
    if h5_confocal_time_lag == 0:
        data_dict['confocal_to_nir'] = confocal_to_nir[:param['max_t']]
        data_dict['timing_stack'] = timing_stack[:param['max_t']]
        data_dict['nir_to_confocal'] = np.array([0.0 if x > param['max_t'] else x for x in nir_to_confocal])
    else:
        data_dict['confocal_to_nir'] = confocal_to_nir[h5_confocal_time_lag:]
        data_dict['timing_stack'] = timing_stack[h5_confocal_time_lag:]
        data_dict['nir_to_confocal'] = np.array([
            0.0 if x < h5_confocal_time_lag + 1 else x - h5_confocal_time_lag 
            for x in nir_to_confocal
        ])
    
    # Get timestamps
    data_dict['nir_timestamps'] = get_timestamps(path_h5)
    vec_to_confocal = lambda vec: nir_vec_to_confocal(vec, data_dict['confocal_to_nir'], param['max_t'])
    
    data_dict['timestamps'] = vec_to_confocal(data_dict['nir_timestamps'])
    data_dict['max_t_nir'] = len(data_dict['nir_to_confocal'])
    
    data_dict['avg_timestep'] = (data_dict['timestamps'][-1] - data_dict['timestamps'][0]) / len(data_dict['timestamps'])
    
    # Pre-confocal timesteps
    data_dict['pre_nir_to_confocal'], data_dict['pre_confocal_to_nir'] = pre_confocal_timesteps(data_dict, param)
    data_dict['max_t'] = param['max_t']
    data_dict['t_range'] = param['t_range']
    
    data_dict['pre_max_t'] = len(data_dict['pre_confocal_to_nir'])
    data_dict['pre_t_range'] = np.arange(1, data_dict['pre_max_t'] + 1)
    
    # Stimulus detection
    with h5py.File(path_h5, 'r') as f:
        stim = f['daqmx_ai'][:, 2]  # Python is 0-indexed
    
    timing_nir = detect_nir_timing_from_file(path_h5)
    thresh = max(np.max(stim) / 2, 0.1)
    nir_mean_times = np.round(np.mean(timing_nir, axis=1)).astype(int)
    stim_to_nir = np.where(stim[nir_mean_times] > thresh)[0]
    
    prev_len = 20
    stim_begin_nir = [t for t in stim_to_nir if not any(t - i in stim_to_nir for i in range(1, prev_len + 1))]
    stim_to_confocal = [int(np.max(data_dict['nir_to_confocal'][:stim_begin_nir[i] + 1])) 
                        for i in range(len(stim_begin_nir))]
    
    data_dict['stim_begin_nir'] = stim_begin_nir
    data_dict['stim_begin_confocal'] = stim_to_confocal


def fill_timeskip(traces: np.ndarray, timestamps: np.ndarray,
                  min_timeskip_length: float = 5, timeskip_step: float = 1,
                  fill_val: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fills in timeskips with multiple datapoints for easier visualization.

    Parameters
    ----------
    traces : np.ndarray
        Traces matrix with timeskips (shape: n_traces x n_timepoints)
    timestamps : np.ndarray
        Timestamps for all data points in the traces matrix
    min_timeskip_length : float, optional
        Minimum difference (in seconds) between adjacent data points to qualify as a timeskip (default: 5)
    timeskip_step : float, optional
        Number of seconds per intermediate data point generated (default: 1)
    fill_val : float, optional
        Value to fill in timeskips (default: 0)

    Returns
    -------
    new_timestamps : np.ndarray
        Timestamps with filled timeskips
    new_traces_matrix : np.ndarray
        Traces matrix with filled timeskips
    """
    time_diffs = np.diff(timestamps)
    timeskips = np.where(time_diffs >= min_timeskip_length)[0]
    
    if len(timeskips) == 0:
        return timestamps, traces
    
    num_timeskips = len(timeskips)
    new_traces = [[] for n in range(traces.shape[0])]
    new_timestamps = []
    prev_timeskip = 0
    
    for timeskip in timeskips:
        num_steps = int(np.floor((timestamps[timeskip + 1] - timestamps[timeskip]) / timeskip_step))
        
        for n in range(traces.shape[0]):
            new_traces[n].extend(traces[n, prev_timeskip:timeskip + 1])
            new_traces[n].extend([fill_val] * num_steps)
        
        new_timestamps.extend(timestamps[prev_timeskip:timeskip + 1])
        new_timestamps.extend([timestamps[timeskip] + t * timeskip_step for t in range(1, num_steps + 1)])
        prev_timeskip = timeskip + 1
    
    # Add remaining data
    for n in range(traces.shape[0]):
        new_traces[n].extend(traces[n, timeskips[-1] + 1:])
    new_timestamps.extend(timestamps[timeskips[-1] + 1:])
    
    new_traces_matrix = np.array(new_traces)
    new_timestamps = np.array(new_timestamps)
    
    return new_timestamps, new_traces_matrix


def fill_timeskip_behavior(behavior: np.ndarray, timestamps: np.ndarray,
                           min_timeskip_length: float = 5, timeskip_step: float = 1,
                           fill_val: float = np.nan) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fills in timeskips for 1D behavioral data.

    Parameters
    ----------
    behavior : np.ndarray
        1D behavioral data array
    timestamps : np.ndarray
        Timestamps
    min_timeskip_length : float, optional
        Minimum timeskip length in seconds (default: 5)
    timeskip_step : float, optional
        Step size for filling (default: 1)
    fill_val : float, optional
        Fill value (default: NaN)

    Returns
    -------
    new_timestamps : np.ndarray
        Timestamps with filled timeskips
    new_behavior : np.ndarray
        Behavior data with filled timeskips
    """
    vec = behavior.reshape(1, -1)
    new_timestamps, new_vec = fill_timeskip(vec, timestamps, 
                                            min_timeskip_length=min_timeskip_length,
                                            timeskip_step=timeskip_step,
                                            fill_val=fill_val)
    return new_timestamps, new_vec[0, :]


def pre_confocal_timesteps(data_dict: Dict, param: Dict) -> Tuple[np.ndarray, List]:
    """
    Computes confocal timesteps backwards in time from the beginning of the confocal recording.

    Parameters
    ----------
    data_dict : dict
        Data dictionary containing timing information
    param : dict
        Parameters dictionary

    Returns
    -------
    pre_nir_to_conf : np.ndarray
        NIR to confocal mapping for pre-recording period
    pre_conf_to_nir : list
        Confocal to NIR mapping for pre-recording period
    """
    step = data_dict['avg_timestep'] * param['FLIR_FPS']
    idx = np.where(data_dict['nir_to_confocal'] == 1)[0][0]
    pre_nir_to_conf = np.zeros(idx)
    n_prev = int(np.floor(idx / step))
    pre_conf_to_nir = []
    offset = idx - int(np.floor(step * n_prev))
    
    for t in range(n_prev):
        rng_start = int(np.floor(t * step + offset))
        rng_end = int(np.floor((t + 1) * step + offset))
        rng = np.arange(rng_start, rng_end)
        pre_nir_to_conf[rng] = t + 1
        pre_conf_to_nir.append(rng)
    
    return pre_nir_to_conf, pre_conf_to_nir
