import h5py
import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import mode
from scipy.signal import find_peaks

def print_h5_structure(h5_path):
    """Recursively print all datasets and groups in an HDF5 file."""
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:40s} shape={str(obj.shape):20s} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  {name}/  (group)")

    with h5py.File(h5_path, 'r') as f:
        print(f"\nHDF5 file structure: {h5_path}\n")
        f.visititems(visitor)
        
        # Key checks
        print("\n--- Key Dataset Info ---")
        if 'img_nir' in f:
            shape = f['img_nir'].shape
            print(f"  img_nir: {shape}  (n_frames={shape[0] if len(shape)==3 else shape[2]})")
        else:
            print("  ⚠ img_nir NOT FOUND")
            
        if 'daqmx_ai' in f:
            print(f"  daqmx_ai: {f['daqmx_ai'].shape}")
        else:
            print("  ⚠ daqmx_ai NOT FOUND")
            
        if 'daqmx_di' in f:
            print(f"  daqmx_di: {f['daqmx_di'].shape}")
        else:
            print("  ⚠ daqmx_di NOT FOUND")
            
        if 'img_metadata' in f:
            for key in f['img_metadata'].keys():
                print(f"  img_metadata/{key}: {f['img_metadata'][key].shape}")
        else:
            print("  ⚠ img_metadata NOT FOUND")
            
            
def load_img_nir(h5_path, frames=None):
    """Load a specific frame of img_nir from the HDF5 file.
    
    Args:
        h5_path (str): Path to the HDF5 file.
        frames (int or list of int, optional): Frame index or indices to load. If set to None, loads all frames.

    Returns:
        np.ndarray: The requested frame(s) of img_nir.

    Raises:
        KeyError: If the img_nir dataset is not found in the HDF5 file.

    """
    with h5py.File(h5_path, 'r') as f:
        if 'img_nir' in f:
            if frames is None:
                img_nir = f['img_nir'][:]
            elif isinstance(frames, int):
                img_nir = f['img_nir'][frames]
            elif isinstance(frames, list):
                img_nir = f['img_nir'][frames]
            else:
                raise ValueError("frames must be an int, list of int, or None")
            return img_nir
        else:
            raise KeyError("img_nir dataset not found in HDF5 file.")


# Assuming 'signal' is your numpy array 
# and 'peaks' is your array of peak indices
def find_wave_starts(signal, peaks):
    starts = []
    
    for i in range(len(peaks)):
        if i == 0:
            # For the first peak, search from the beginning of the signal
            start_idx = np.argmin(signal[:peaks[i]])
        else:
            # Search between the previous peak and current peak
            search_range = signal[peaks[i-1] : peaks[i]]
            # Add the offset of the previous peak to get the absolute index
            start_idx = np.argmin(search_range) + peaks[i-1]
            
        starts.append(start_idx)
        
    return np.array(starts)

# Example usage:
# start_indices = find_wave_starts(my_signal, peak_indices)

def map_timestamps(source_timing, reference_timing):
    """
    Finds which interval in reference_timing each source_start falls into.
    Assumes reference_timing is sorted by start times.
    """
    # Extract start times for binary search
    ref_starts = reference_timing[:, 0]
    ref_stops = reference_timing[:, 1]
    source_starts = source_timing[:, 0]

    # Find the index where each source_start would fit in ref_starts
    # 'side=right' - 1 gives us the index j such that ref_starts[j] <= source_start
    idx = np.searchsorted(ref_starts, source_starts, side='right') - 1

    # Validation: Check if it actually falls within the [start, stop] window
    # If the index is -1 or source_start > stop, it doesn't fit any interval
    mask = (idx >= 0) & (source_starts <= ref_stops[idx])
    
    # Create result array (NaN for no match, 0-based index of matching interval otherwise)
    result = np.full(source_starts.shape[0], np.nan)
    result[mask] = idx[mask]

    return result

def generate_consecutive_counts(arr):
    # # Example usage:
    # arr = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 5, 5, 5, 5])
    # print(generate_consecutive_counts(arr))
    # # Output: [0 1 2 3 4 0 1 2 3 4 0 1 2 3]
    
    # Ensure input is a numpy array
    arr = np.asarray(arr)
    
    # Identify where the values change (the first element is always a change)
    # np.diff returns an array of differences between consecutive elements
    # We check where that difference is not zero
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    
    # Create an array of zeros to store the result
    res = np.zeros(len(arr), dtype=int)
    
    # The starting positions of each new group are at index 0 and 
    # all positions identified by change_indices
    starts = np.concatenate(([0], change_indices))
    
    # For each group, we subtract the start index from the current index
    # to get the running count: [0, 1, 2, ...]
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i+1] if i + 1 < len(starts) else len(arr)
        res[start:end] = np.arange(end - start)
        
    return res



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


def detect_confocal_timing_di(di_confocal: np.ndarray
                           ) -> np.ndarray:
    """Detect confocal z-stack start/stop times from the digital input of the confocal camera.

    Parameters
    ----------
    di_confocal : 1-D array
        Digital input from confocal camera.

    Returns
    -------
    np.ndarray — sample indices for each z-stack, with start and stop times.
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

    return np.column_stack((list_stack_start[idx_vol], list_stack_stop[idx_vol]))

def detect_piezo_timing(ai_piezo: np.ndarray, confocal_start_sample_di: int, confocal_stop_sample_di: int) -> np.ndarray:
    
    peaks, properties = find_peaks(
        ai_piezo, 
        distance=100,
        prominence=1.0
    ) # find peaks in piezo signal, which should correspond to stack acquisition. distance is set to 400 samples
    stack_start_sample = peaks.copy()
    stack_start_sample = np.insert(stack_start_sample, 0, confocal_start_sample_di)  # prepend first element as confocal recording start
    stack_start_sample = stack_start_sample[:-1]  # delete last element

    stack_stop_sample = peaks.copy()
    stack_stop_sample[-1] = confocal_stop_sample_di # set last stack stop to confocal recording stop
    piezo_timing = np.concatenate((stack_start_sample.reshape(-1,1), stack_stop_sample.reshape(-1,1)), axis=1) # combine piezo start and stop indices to get confocal stack timing
    
    return piezo_timing

# ---------------------------------------------------------------------------
# sync_timing
# ---------------------------------------------------------------------------

def sync_timing(di_confocal: np.ndarray,
                ai_laser: np.ndarray,
                di_nir: np.ndarray,
                ai_piezo: np.ndarray,
                img_id: np.ndarray,
                q_iter_save: np.ndarray,
                n_img_nir: int,
                img_timestamp: np.ndarray) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synchronize confocal and NIR timing.

    Parameters
    ----------
    di_confocal : 1-D array
        Digital trigger from confocal microscope.
    ai_laser : 1-D array
        Filtered analog laser signal (from ``filter_ai_laser``).
    di_nir : 1-D array
        Digital trigger from NIR camera.
    ai_piezo : 1-D array
        Piezo analog signal.

    Returns
    -------
    confocal_to_nir : list[np.ndarray]
        For each confocal volume, the indices of NIR frames within it.
    nir_to_confocal : 1-D float array
        For each NIR frame, the **1-based** confocal volume index.
        NaN for NIR frames with no corresponding confocal volume
        (before/after confocal recording or during gaps).
    timing_stack : (n_frames, 2) array
        Confocal frame ``[start, stop]`` sample indices.
    timing_nir : (n_nir, 2) array
        NIR frame ``[on, off]`` sample indices.
    timing_piezo : (n_volumes, 2) array
        Piezo ``[start, stop]`` sample indices.
    nir_time_sec : 1-D array
        NIR frame timestamps in seconds.
    piezo_to_confocal : 1-D float array
        For each confocal DI event (row of timing_confocal), the 0-based
        piezo volume it belongs to.  NaN if it falls outside all piezo
        intervals.
    """
    timing_confocal = np.column_stack(detect_confocal_timing_di(di_confocal)).T # confocal frame start stop times from digital input, shape (n_frames, 2)
    timing_nir   = detect_nir_timing(di_nir, img_id, q_iter_save, n_img_nir) # NIR frame on off times from digital input, shape (n_nir, 2)
    timing_piezo = detect_piezo_timing(ai_piezo, timing_confocal[0, 0], timing_confocal[-1, 1]) # piezo start stop times from analog input, shape (n_volumes, 2)

    # Map each confocal DI event to its piezo volume (0-based, NaN if outside)
    piezo_to_confocal = map_timestamps(timing_confocal, timing_piezo) # for each confocal frame, which piezo volume (i.e. confocal stack) it belongs to (0-based index, NaN if outside all volumes)

    confocal_to_nir  = []
    # NaN by default: NIR frames before/after confocal recording or
    # during gaps between piezo volumes will remain NaN.
    nir_to_confocal  = np.full(timing_nir.shape[0], np.nan)

    for i in range(timing_piezo.shape[0]):
        start_, end_ = timing_piezo[i, :]

        nir_on_in  = (start_ < timing_nir[:, 0]) & (timing_nir[:, 0] < end_)
        nir_off_in = (start_ < timing_nir[:, 1]) & (timing_nir[:, 1] < end_)

        idx_ = np.where(nir_on_in & nir_off_in)[0]
        confocal_to_nir.append(idx_)
        nir_to_confocal[idx_] = i            # 0-based
        
    nir_time_sec = ((img_timestamp[q_iter_save] - img_timestamp[q_iter_save][0]) + timing_nir[0,0]) / 1e9 # convert to seconds and set first timestamp to first nir trigger

    return confocal_to_nir, nir_to_confocal, timing_confocal, timing_nir, timing_piezo, nir_time_sec, piezo_to_confocal